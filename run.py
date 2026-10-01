#!/usr/bin/env python3
"""Run one multiclass canonical-lift PAC-Bayes experiment."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.distributed as dist

from data import fit_A_only_input_transform, load_test_set, load_training_set, observation_independent_split
from models import (
    CanonicalPosterior,
    PriorModel, apply_feature_transform, extract_prior_outputs, feature_transform_report,
    fit_feature_transform, load_upstream_feature_transform, make_backbone,
    make_random_feature_transform,
    train_or_load_prior, validate_upstream_backbone,
)
from pac_bayes import (
    fresh_monte_carlo, gauss_hermite_risk,
    kl_upper_inverse, observable_coordinates, optimize_posterior, pac_bayes_certificate,
)
from utils import (
    choose_device, json_ready, load_config, runtime_info, save_json,
    set_deterministic, smoke_config,
)


HERE = Path(__file__).resolve().parent
PRESETS = HERE / "configs"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--preset", choices=sorted(path.stem for path in PRESETS.glob("*.json")))
    configuration.add_argument("--config", type=Path)
    configuration.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--save-prior-checkpoint", type=Path)
    parser.add_argument("--upstream-stats", type=Path)
    parser.add_argument("--encoder-weights", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()

    # 1. Parse arguments and load the scientific configuration.
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.smoke:
        config = smoke_config()
    elif args.preset:
        config = load_config(PRESETS / f"{args.preset}.json")
    else:
        config = load_config(args.config)
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    config["seed"] = args.seed
    config["dataset"]["split_seed"] = args.seed
    config["posterior"]["seed"] = args.seed + 10_000
    config["certification"]["monte_carlo_seed"] = args.seed + 20_000
    device = choose_device(args.device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed else 0
    set_deterministic(config["seed"])
    data_root, output_path = args.data_root.resolve(), args.output.resolve()
    if rank == 0:
        print(f"[setup] {config['name']} on {device}")

    # A random prior is fully fixed before reading downstream data.
    if config["prior"]["source"] == "random":
        if any(value is not None for value in (
            args.prior_checkpoint, args.upstream_stats, args.encoder_weights
        )):
            raise ValueError("a random prior does not accept checkpoint, artifact, or encoder weights")
        torch.manual_seed(config["encoder"]["initialization_seed"])
        backbone = make_backbone(
            config["encoder"], config["dataset"].get("synthetic_input_dimension"), None
        )
        prior = PriorModel(backbone, config["dataset"]["number_classes"], zero_head=True)
        prior.freeze()
        input_transform = {"kind": "fixed_minus_one_one"}
        feature_transform = make_random_feature_transform(
            backbone.feature_dim, config["feature_map"]
        )
        torch.manual_seed(config["seed"])

    # 2. Load only the training dataset. Test data is deliberately not loaded here.
    train_images, train_labels = load_training_set(
        config["dataset"], data_root, args.download, config["seed"]
    )

    # 3. Split indices into A and B without giving the split function any observations.
    A_indices, B_indices = observation_independent_split(
        train_labels.numel(), config["dataset"]["prior_fraction"], config["dataset"]["split_seed"]
    )
    if rank == 0:
        print(f"[data] A={A_indices.numel()} B={B_indices.numel()} (index-only split)")

    # 4. Train or load the deterministic prior. Every learned choice here uses A only.
    upstream_audit = None
    if config["prior"]["source"] == "a_trained":
        backbone = make_backbone(
            config["encoder"], config["dataset"].get("synthetic_input_dimension"),
            args.encoder_weights.resolve() if args.encoder_weights else None,
        )
        if args.upstream_stats is not None:
            raise ValueError("--upstream-stats is only valid for the transfer preset")
        input_transform = fit_A_only_input_transform(
            train_images, A_indices, config["dataset"]["name"]
        )
        prior = PriorModel(backbone, config["dataset"]["number_classes"], zero_head=False)
    elif config["prior"]["source"] == "upstream":
        backbone = make_backbone(
            config["encoder"], config["dataset"].get("synthetic_input_dimension"),
            args.encoder_weights.resolve() if args.encoder_weights else None,
        )
        if args.upstream_stats is None:
            raise ValueError("the transfer preset requires --upstream-stats")
        if args.prior_checkpoint is not None:
            raise ValueError("the transfer preset does not use --prior-checkpoint")
        feature_transform, input_transform, upstream_audit = load_upstream_feature_transform(
            args.upstream_stats.resolve(), config["feature_map"]
        )
        validate_upstream_backbone(backbone)
        prior = PriorModel(backbone, config["dataset"]["number_classes"], zero_head=True)

    prior_metrics = train_or_load_prior(
        prior, train_images, train_labels, A_indices, input_transform, config["prior"],
        device, args.workers, config["seed"],
        args.prior_checkpoint.resolve() if args.prior_checkpoint else None,
    )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
        if rank != 0:
            return {"worker_rank": rank, "status": "prior_training_complete"}
    if args.save_prior_checkpoint is not None:
        if config["prior"]["source"] != "a_trained":
            raise ValueError("only an A-trained prior can be saved as a checkpoint")
        checkpoint_path = args.save_prior_checkpoint.resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {
                name: value.detach().cpu() for name, value in prior.state_dict().items()
            },
            "metadata": {
                "number_classes": config["dataset"]["number_classes"],
                "A_indices": A_indices.clone(),
                "selected_epoch": prior_metrics["selected_epoch"],
                "seed": config["seed"],
            },
        }, checkpoint_path)
        print(f"[prior] saved {checkpoint_path}")
    print(f"[prior] epoch={prior_metrics['selected_epoch']} training_gpus={prior_metrics['training_gpu_count']}")

    # 5. Fit the stochastic feature transformation using A only.
    extraction_dtype = (
        torch.float32 if config["dataset"]["name"] == "imagenet" else torch.float64
    )
    if config["prior"]["source"] == "a_trained":
        raw_A, _, _ = extract_prior_outputs(
            prior, train_images, train_labels, A_indices, input_transform,
            max(config["prior"]["batch_size"], 256), device, args.workers,
            include_scores=False, output_dtype=extraction_dtype,
        )
        feature_transform = fit_feature_transform(raw_A, config["feature_map"])
        del raw_A

    # 6. Freeze the prior, then extract frozen base scores and stochastic features.
    prior.freeze()
    if any(parameter.requires_grad for parameter in prior.parameters()):
        raise RuntimeError("the prior was not frozen")
    raw_B, base_scores_B, labels_B = extract_prior_outputs(
        prior, train_images, train_labels, B_indices, input_transform,
        max(config["prior"]["batch_size"], 256), device, args.workers,
        output_dtype=extraction_dtype,
    )
    features_B = apply_feature_transform(raw_B, feature_transform)
    del raw_B
    base_scores_B = base_scores_B.to(torch.float64)
    if config["prior"]["source"] == "random":
        if bool(torch.any(features_B.square().sum(1) == 0.0)):
            raise RuntimeError("random feature map lost its required constant coordinate")
        prior_metrics["exact_population_gibbs_risk"] = (
            1.0 - 1.0 / config["dataset"]["number_classes"]
        )

    # 7. Compute B's observable coordinates, retaining every nonzero SVD direction.
    coordinates = observable_coordinates(
        features_B, config["numerics"]["minimum_relative_singular_value"]
    )
    if coordinates["observed_rank"] != config["feature_map"]["rank"]:
        raise RuntimeError("observable B rank differs from the declared feature rank")
    if config["prior"]["source"] == "random":
        # The SVD is an audit only. The diagonal posterior stays in fixed phi_R coordinates.
        coordinates["right_basis"] = torch.eye(features_B.shape[1], dtype=torch.float64)
        coordinates["posterior_coordinate_system"] = "fixed_random_feature_coordinates"
    print(
        f"[support] rank={coordinates['observed_rank']}/{coordinates['latent_dimension']} "
        f"condition={coordinates['condition_number']:.6g} discarded=0"
    )

    # 8. Optimize and select the posterior using B only (Q=P remains a candidate).
    posterior, selected, q_equals_p, states_evaluated = optimize_posterior(
        config["posterior"], coordinates, features_B, base_scores_B, labels_B,
        config["dataset"]["number_classes"], config["numerics"]["minimum_posterior_std"],
        config["confidence"]["pac_bayes_delta_each"], device,
    )

    # 9. Freeze the selected posterior before any reported KL or certification draw.
    posterior.freeze()
    if any(parameter.requires_grad for parameter in posterior.parameters()):
        raise RuntimeError("the selected posterior was not frozen")

    # 10. Compute raw and quotient KL from that exact same frozen posterior law.
    raw_kl_tensor, quotient_kl_tensor = posterior.kl_values()
    raw_kl_value, quotient_kl_value = float(raw_kl_tensor), float(quotient_kl_tensor)
    print(
        f"[posterior] {selected['candidate']} step={selected['step']} "
        f"raw_kl={raw_kl_value:.8g} quotient_kl={quotient_kl_value:.8g}"
    )

    direct_holdout = None
    if config["certification"]["direct_holdout"] and config["prior"]["source"] == "a_trained":
        # Q=P is the stochastic prior. MC bounds its empirical Gibbs risk on B;
        # a fixed-function binary-KL step adds concentration from B to population.
        direct_total_delta = config["confidence"]["direct_holdout_delta"]
        direct_mc_delta = config["confidence"]["monte_carlo_delta"]
        direct_data_delta = direct_total_delta - direct_mc_delta
        if direct_data_delta <= 0.0:
            raise ValueError("direct holdout delta must exceed its MC allocation")
        stochastic_prior = CanonicalPosterior(
            coordinates["right_basis"], config["dataset"]["number_classes"],
            config["numerics"]["minimum_posterior_std"],
        )
        stochastic_prior.freeze()
        prior_mc = fresh_monte_carlo(
            stochastic_prior, features_B, base_scores_B, labels_B,
            config["certification"]["monte_carlo_trials"],
            config["certification"]["monte_carlo_chunk_size"],
            config["certification"]["monte_carlo_seed"] + 1,
            direct_mc_delta,
        )
        direct_budget = math.log(1.0 / direct_data_delta) / labels_B.numel()
        direct_upper = kl_upper_inverse(
            prior_mc["clopper_pearson_upper"], direct_budget
        )
        direct_holdout = {
            "method": "stochastic_prior_Q_equals_P",
            "sample_size": labels_B.numel(),
            "monte_carlo": prior_mc,
            "monte_carlo_delta": direct_mc_delta,
            "concentration_delta": direct_data_delta,
            "total_delta": direct_total_delta,
            "concentration_binary_kl_budget": direct_budget,
            "concentration_increase": (
                direct_upper - prior_mc["clopper_pearson_upper"]
            ),
            "population_gibbs_risk_upper": direct_upper,
            "upper": direct_upper,
            "confidence_statement_is_separate": True,
        }

    # 11. Use a new, separately seeded stream for post-selection Monte Carlo.
    fresh_mc = fresh_monte_carlo(
        posterior, features_B, base_scores_B, labels_B,
        config["certification"]["monte_carlo_trials"],
        config["certification"]["monte_carlo_chunk_size"],
        config["certification"]["monte_carlo_seed"],
        config["confidence"]["monte_carlo_delta"],
    )

    # 12. Combine the same conservative MC endpoint with output and raw KL.
    certificate = pac_bayes_certificate(
        fresh_mc["clopper_pearson_upper"], quotient_kl_value, labels_B.numel(),
        config["confidence"]["pac_bayes_delta_each"],
    )
    raw_gaussian_certificate = pac_bayes_certificate(
        fresh_mc["clopper_pearson_upper"], raw_kl_value, labels_B.numel(),
        config["confidence"]["pac_bayes_delta_each"],
    )
    confidence_report = {
        "n_B": labels_B.numel(),
        "pac_bayes_delta_each": config["confidence"]["pac_bayes_delta_each"],
        "pac_bayes_family_count": config["confidence"]["pac_bayes_family_count"],
        "monte_carlo_delta": config["confidence"]["monte_carlo_delta"],
        "joint_failure_allocation": (
            config["confidence"]["pac_bayes_delta_each"] * config["confidence"]["pac_bayes_family_count"]
            + config["confidence"]["monte_carlo_delta"]
        ),
        "selection_preceded_final_mc": True,
    }
    certificate.update(confidence_report)
    certificate["kl_type"] = "output_space_quotient"
    raw_gaussian_certificate.update(confidence_report)
    raw_gaussian_certificate["kl_type"] = "raw_gaussian_parameter"
    print(
        f"[certificate] MC={100*fresh_mc['observed_risk']:.4f}% "
        f"CP={100*fresh_mc['clopper_pearson_upper']:.4f}% "
        f"bound={100*certificate['population_gibbs_risk_upper']:.4f}%"
    )

    # 13. Only now load and evaluate the test set; these values are diagnostic only.
    diagnostics: dict[str, Any] = {"test_role": "diagnostic_only", "test_loaded_after_certificate": False}
    if config["certification"]["evaluate_test"]:
        test_images, test_labels = load_test_set(config["dataset"], data_root, args.download, config["seed"])
        test_indices = torch.arange(test_labels.numel(), dtype=torch.long)
        raw_test, base_scores_test, labels_test = extract_prior_outputs(
            prior, test_images, test_labels, test_indices, input_transform,
            max(config["prior"]["batch_size"], 256), device, args.workers,
            output_dtype=extraction_dtype,
        )
        features_test = apply_feature_transform(raw_test, feature_transform)
        del raw_test
        base_scores_test = base_scores_test.to(torch.float64)
        test_risk = gauss_hermite_risk(
            posterior, features_test, base_scores_test, labels_test,
            config["certification"]["diagnostic_quadrature_order"],
            config["posterior"]["selection_chunk_size"],
        )
        mean_test_scores, _ = posterior.score_statistics(features_test, base_scores_test)
        diagnostics.update({
            "test_loaded_after_certificate": True, "test_size": labels_test.numel(),
            "test_gibbs_risk_gauss_hermite": test_risk,
            "test_mean_argmax_error": float((mean_test_scores.argmax(1) != labels_test).to(torch.float64).mean()),
        })

    # 14. Save exactly one compact JSON result.
    paper_table = {
        "ours_output_space_kl": {
            "R_hat_B_percent": 100.0 * fresh_mc["observed_risk"],
            "KL_over_n_B": quotient_kl_value / labels_B.numel(),
            "certificate_percent": 100.0 * certificate["population_gibbs_risk_upper"],
            "test_risk_percent": (
                100.0 * diagnostics["test_gibbs_risk_gauss_hermite"]
                if diagnostics["test_loaded_after_certificate"] else None
            ),
        },
        "raw_gaussian_parameter_kl": {
            "R_hat_B_percent": 100.0 * fresh_mc["observed_risk"],
            "KL_over_n_B": raw_kl_value / labels_B.numel(),
            "certificate_percent": 100.0 * raw_gaussian_certificate["population_gibbs_risk_upper"],
            "test_risk_percent": (
                100.0 * diagnostics["test_gibbs_risk_gauss_hermite"]
                if diagnostics["test_loaded_after_certificate"] else None
            ),
        },
        "direct_holdout_stochastic_prior": (
            None if direct_holdout is None else {
                "R_hat_B_percent": 100.0 * direct_holdout["monte_carlo"]["observed_risk"],
                "KL_over_n_B": 0.0,
                "certificate_percent": 100.0 * direct_holdout["population_gibbs_risk_upper"],
                "test_risk_percent": None,
            }
        ),
    }
    metrics = {
        "status": "certified",
        "data": {
            "dataset": config["dataset"]["name"], "training_size": train_labels.numel(),
            "A_size": A_indices.numel(), "B_size": B_indices.numel(),
            "observation_independent_unstratified_split": True,
        },
        "prior": {
            **prior_metrics, "input_transform": input_transform,
            "feature_transform": feature_transform_report(feature_transform),
            "direct_holdout": direct_holdout, "upstream": upstream_audit,
        },
        "support": {key: value for key, value in coordinates.items() if key not in {"right_basis", "singular_values"}},
        "posterior": {
            **selected,
            "q_equals_p_B_gauss_hermite_risk": q_equals_p["B_gauss_hermite_risk"],
            "q_equals_p_selection_upper": q_equals_p["selection_upper"],
            "candidate_states_evaluated": states_evaluated,
        },
        "kl": {
            "raw_nats": raw_kl_value, "quotient_nats": quotient_kl_value,
        },
        "fresh_mc": fresh_mc, "certificate": certificate,
        "raw_gaussian_certificate": raw_gaussian_certificate,
        "paper_table": paper_table, "diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result = {
        "config": {
            "scientific": config, "runtime": runtime_info(device),
        },
        "metrics": metrics,
        "report": (
            f"{config['name']}: certified Gibbs-risk upper bound "
            f"{100*certificate['population_gibbs_risk_upper']:.4f}% "
            f"(n_B={labels_B.numel()}, quotient KL={quotient_kl_value:.6g} nats)."
        ),
    }
    save_json(output_path, json_ready(result))
    print(f"[done] wrote {output_path}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
