"""Observable coordinates, posterior fitting, exact KL bounds, and final MC."""

from __future__ import annotations

from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, ROUND_CEILING, ROUND_FLOOR, getcontext
from functools import lru_cache
import math
import operator
import struct
from typing import Any, Mapping

from scipy.special import betaincinv
import torch
from torch import Tensor
from torch.nn import functional as F

from models import CanonicalPosterior


# -----------------------------------------------------------------------------
# Every nonzero B direction is retained. The threshold audits conditioning only.


def observable_coordinates(features_B: Tensor, minimum_relative: float) -> dict[str, Any]:
    values = features_B.detach().cpu().to(torch.float64)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("B features must have nonempty shape [n,r]")
    if values.shape[0] < values.shape[1]:
        raise RuntimeError("the protocol requires n_B >= feature dimension")
    _, singular_values, right_transpose = torch.linalg.svd(values, full_matrices=False)
    if singular_values.numel() != values.shape[1]:
        raise RuntimeError("compact SVD did not return every feature direction")
    if not bool(torch.all(torch.isfinite(singular_values))) or float(singular_values[-1]) <= 0.0:
        raise RuntimeError("B feature matrix is rank deficient; no direction may be truncated")
    relative = float(singular_values[-1] / singular_values[0])
    if relative < minimum_relative:
        raise RuntimeError("B feature matrix failed its conditioning audit; no direction was discarded")
    right_basis = right_transpose.T.contiguous()
    identity = torch.eye(right_basis.shape[1], dtype=torch.float64)
    if not torch.allclose(right_basis.T @ right_basis, identity, atol=2e-12, rtol=2e-12):
        raise RuntimeError("SVD right basis is not numerically orthonormal")
    return {
        "right_basis": right_basis,
        "singular_values": singular_values,
        "number_examples": values.shape[0],
        "latent_dimension": values.shape[1],
        "observed_rank": singular_values.numel(),
        "directions_discarded": 0,
        "smallest_singular_value": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }


# -----------------------------------------------------------------------------
# Exact one-dimensional Gaussian integration of independent multiclass scores.


def independent_score_errors_gauss_hermite(
    means: Tensor,
    stds: Tensor,
    labels: Tensor,
    order: int,
    quadrature_chunk_size: int = 256,
    class_chunk_size: int = 128,
) -> Tensor:
    if means.ndim != 2 or means.shape[1] < 2 or means.shape[0] == 0:
        raise ValueError("means must have shape [n,K>=2]")
    if stds.shape != means.shape or bool(torch.any(stds <= 0.0)):
        raise ValueError("standard deviations must be positive and match means")
    if labels.shape != (means.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be one-dimensional torch.long")
    nodes_cpu, log_weights_cpu = _standard_normal_rule(order)
    nodes = nodes_cpu.to(device=means.device, dtype=means.dtype)
    log_weights = log_weights_cpu.to(device=means.device, dtype=means.dtype)
    true_means = means.gather(1, labels[:, None]).squeeze(1)
    true_stds = stds.gather(1, labels[:, None]).squeeze(1)
    log_total = None
    for point_start in range(0, order, quadrature_chunk_size):
        point_end = min(point_start + quadrature_chunk_size, order)
        selected_nodes = nodes[point_start:point_end]
        sampled_true = true_means[:, None] + true_stds[:, None] * selected_nodes[None, :]
        log_integrand = means.new_zeros((means.shape[0], selected_nodes.numel()))
        for class_start in range(0, means.shape[1], class_chunk_size):
            class_end = min(class_start + class_chunk_size, means.shape[1])
            arguments = (
                sampled_true[:, :, None] - means[:, None, class_start:class_end]
            ) / stds[:, None, class_start:class_end]
            # factors = torch.special.log_ndtr(arguments)
            if arguments.device.type == "mps":
                original_device = arguments.device
                original_dtype = arguments.dtype

                factors = torch.special.log_ndtr(
                    arguments.to(device="cpu", dtype=torch.float32)
                ).to(device=original_device, dtype=original_dtype)
            else:
                factors = torch.special.log_ndtr(arguments)
            class_indices = torch.arange(class_start, class_end, device=means.device)
            is_true = labels[:, None] == class_indices[None, :]
            factors = torch.where(is_true[:, None, :], torch.zeros_like(factors), factors)
            log_integrand += factors.sum(2)
        chunk = torch.logsumexp(
            log_integrand + log_weights[point_start:point_end][None, :], dim=1
        )
        log_total = chunk if log_total is None else torch.logaddexp(log_total, chunk)
    return -torch.expm1(log_total.clamp_max(0.0))


@lru_cache(maxsize=32)
def _standard_normal_rule(order: int) -> tuple[Tensor, Tensor]:
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    diagonal = torch.zeros(order, dtype=torch.float64)
    if order == 1:
        return diagonal, diagonal
    off_diagonal = torch.sqrt(torch.arange(1, order, dtype=torch.float64))
    jacobi = torch.diag(diagonal)
    jacobi += torch.diag(off_diagonal, diagonal=1)
    jacobi += torch.diag(off_diagonal, diagonal=-1)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    return nodes, torch.log(eigenvectors[0].square())


def posterior_errors(
    posterior: CanonicalPosterior,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    order: int,
) -> Tensor:
    means, variances = posterior.score_statistics(features, base_scores)
    stochastic = torch.all(variances > 0.0, dim=1)
    deterministic = torch.all(variances == 0.0, dim=1)
    if not bool(torch.all(stochastic | deterministic)):
        raise RuntimeError("each example must be wholly stochastic or deterministic")
    safe_stds = torch.sqrt(torch.where(stochastic[:, None], variances, torch.ones_like(variances)))
    stochastic_errors = independent_score_errors_gauss_hermite(means, safe_stds, labels, order)
    deterministic_errors = (means.argmax(1) != labels).to(means.dtype)
    return torch.where(stochastic, stochastic_errors, deterministic_errors)


@torch.inference_mode()
def gauss_hermite_risk(
    posterior: CanonicalPosterior,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    order: int,
    chunk_size: int,
) -> float:
    total = 0.0
    for start in range(0, labels.numel(), chunk_size):
        stop = min(start + chunk_size, labels.numel())
        total += float(posterior_errors(
            posterior, features[start:stop], base_scores[start:stop], labels[start:stop], order
        ).sum())
    return total / labels.numel()


# -----------------------------------------------------------------------------
# PAC-Bayes-kl certificate and the three paper optimization surrogates.


def binary_kl(empirical: float, candidate: float) -> float:
    if empirical == candidate:
        return 0.0
    if empirical == 0.0:
        return math.inf if candidate == 1.0 else -math.log1p(-candidate)
    if empirical == 1.0:
        return math.inf if candidate == 0.0 else -math.log(candidate)
    if candidate in {0.0, 1.0}:
        return math.inf
    return empirical * math.log(empirical / candidate) + (1.0 - empirical) * math.log(
        (1.0 - empirical) / (1.0 - candidate)
    )


def kl_upper_inverse(empirical: float, budget: float, iterations: int = 80) -> float:
    if empirical == 1.0 or math.isinf(budget):
        return 1.0
    if budget == 0.0:
        return empirical
    if empirical == 0.0:
        return min(1.0, -math.expm1(-budget))
    lower, upper = empirical, 1.0
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if binary_kl(empirical, midpoint) > budget:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def pac_bayes_certificate(
    empirical_risk: float, output_kl: float, sample_size: int, delta: float
) -> dict[str, float]:
    constant = math.log(2.0) + 0.5 * math.log(sample_size) - math.log(delta)
    budget = (max(0.0, output_kl) + constant) / sample_size
    return {
        "empirical": empirical_risk,
        "binary_kl_budget": budget,
        "population_gibbs_risk_upper": kl_upper_inverse(empirical_risk, budget),
    }


def training_objective(
    name: str, empirical_risk: Tensor, output_kl: Tensor, sample_size: int, delta: float
) -> Tensor:
    constant = output_kl.new_tensor(
        math.log(2.0) + 0.5 * math.log(sample_size) - math.log(delta)
    )
    if name == "classic":
        return empirical_risk + torch.sqrt(
            (output_kl.clamp_min(0.0) + constant) / (2.0 * sample_size)
        )
    complexity = (output_kl.clamp_min(0.0) + constant) / (2.0 * sample_size)
    if name == "quad":
        return (torch.sqrt(empirical_risk + complexity) + torch.sqrt(complexity)).square()
    if name == "exact":
        budget = (output_kl.clamp_min(0.0) + constant) / sample_size
        return _DifferentiableKLUpperInverse.apply(empirical_risk, budget)
    raise ValueError(f"unknown objective: {name}")


class _DifferentiableKLUpperInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, empirical: Tensor, budget: Tensor) -> Tensor:
        upper = empirical.new_tensor(kl_upper_inverse(float(empirical.detach()), float(budget.detach())))
        if empirical.device.type == "mps":
            ctx.save_for_backward(empirical.detach().to(torch.float32), upper.detach().to(torch.float32))
        else:
            ctx.save_for_backward(empirical.detach().to(torch.float64), upper.detach().to(torch.float64))
        return upper

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, Tensor]:
        empirical, upper = ctx.saved_tensors
        epsilon = torch.finfo(torch.float64).eps
        p = empirical.clamp(epsilon, 1.0 - epsilon)
        q = upper.clamp(epsilon, 1.0 - epsilon)
        derivative_q = (q - p) / (q * (1.0 - q))
        derivative_p = torch.log(p) - torch.log1p(-p) - torch.log(q) + torch.log1p(-q)
        return gradient * (-derivative_p / derivative_q).to(gradient), gradient * (1.0 / derivative_q).to(gradient)


# -----------------------------------------------------------------------------
# B-only posterior optimization and selection on common CPU-float64 evaluations.


def optimize_posterior(
    config: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    features_B: Tensor,
    base_scores_B: Tensor,
    labels_B: Tensor,
    number_classes: int,
    minimum_std: float,
    delta: float,
    device: torch.device,
) -> tuple[CanonicalPosterior, dict[str, Any], dict[str, Any], int]:
    if features_B.dtype != torch.float64 or features_B.device.type != "cpu":
        raise ValueError("posterior selection inputs must be CPU float64")
    basis = coordinates["right_basis"]
    # Large-class ImageNet runs use a fixed B subset only to rank checkpoints.
    # This score is not reported as the certificate: fresh MC later evaluates
    # the frozen winner over all of B.
    selection_count = min(config.get("selection_examples", labels_B.numel()), labels_B.numel())
    selection_indices = torch.randperm(
        labels_B.numel(), generator=torch.Generator().manual_seed(config["seed"] + 777)
    )[:selection_count]
    selection_features = features_B.index_select(0, selection_indices)
    selection_scores = base_scores_B.index_select(0, selection_indices)
    selection_labels = labels_B.index_select(0, selection_indices)
    initial = CanonicalPosterior(basis, number_classes, minimum_std)
    q_equals_p = _posterior_summary(
        "Q=P", None, None, 0, initial,
        selection_features, selection_scores, selection_labels,
        config, delta, labels_B.numel(),
    )
    selected = q_equals_p
    selected_state = _cpu_state(initial.state_dict())
    states_evaluated = 1

    training_dtype = torch.float64 if device.type == "cpu" else torch.float32
    training_basis = basis.to(device=device, dtype=training_dtype)
    training_features = features_B.to(device=device, dtype=training_dtype)
    training_scores = base_scores_B.to(device=device, dtype=training_dtype)
    training_labels = labels_B.to(device=device)

    for objective_index, objective_name in enumerate(config["objectives"]):
        for rate_index, learning_rate in enumerate(config["learning_rates"]):
            candidate = CanonicalPosterior(training_basis, number_classes, minimum_std)
            optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
            generator = torch.Generator(device="cpu").manual_seed(
                config["seed"] + 10_000 * objective_index + 101 * rate_index
            )
            for _ in range(config["warmup_steps"]):
                indices = _sample_indices(labels_B.numel(), config["batch_size"], generator, device)
                projected = training_features.index_select(0, indices) @ candidate.right_basis
                scores = training_scores.index_select(0, indices) + projected @ candidate.mean.T
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(scores, training_labels.index_select(0, indices))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), config["gradient_clip_norm"])
                optimizer.step()
                candidate.remove_common_mean()

            for step in range(1, config["steps"] + 1):
                indices = _sample_indices(labels_B.numel(), config["batch_size"], generator, device)
                optimizer.zero_grad(set_to_none=True)
                empirical = posterior_errors(
                    candidate,
                    training_features.index_select(0, indices),
                    training_scores.index_select(0, indices),
                    training_labels.index_select(0, indices),
                    config["training_quadrature_order"],
                ).mean()
                _, output_kl = candidate.kl_values()
                objective = training_objective(
                    objective_name, empirical, output_kl, labels_B.numel(), delta
                )
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError("posterior objective became non-finite")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), config["gradient_clip_norm"])
                optimizer.step()
                candidate.remove_common_mean()

                if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                    portable = CanonicalPosterior(basis, number_classes, minimum_std)
                    portable_state = {
                        name: value.detach().cpu().to(torch.float64)
                        for name, value in candidate.state_dict().items()
                    }
                    # The posterior law uses the exact CPU-float64 B coordinates;
                    # only the fitted mean/std parameters come back from the GPU.
                    portable_state["right_basis"] = basis
                    portable.load_state_dict(portable_state, strict=True)
                    summary = _posterior_summary(
                        f"{objective_name} lr={learning_rate:g}", objective_name, learning_rate,
                        step, portable, selection_features, selection_scores, selection_labels,
                        config, delta, labels_B.numel(),
                    )
                    states_evaluated += 1
                    if _selection_key(summary) < _selection_key(selected):
                        selected = summary
                        selected_state = _cpu_state(portable.state_dict())

    posterior = CanonicalPosterior(basis, number_classes, minimum_std)
    posterior.load_state_dict(selected_state, strict=True)
    replay = _posterior_summary(
        selected["candidate"], selected["objective"], selected["learning_rate"], selected["step"],
        posterior, selection_features, selection_scores, selection_labels,
        config, delta, labels_B.numel(),
    )
    if replay != selected:
        raise RuntimeError("selected posterior did not replay exactly")
    return posterior, selected, q_equals_p, states_evaluated


def _posterior_summary(
    candidate_name: str,
    objective_name: str | None,
    learning_rate: float | None,
    step: int,
    posterior: CanonicalPosterior,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    config: Mapping[str, Any],
    delta: float,
    complexity_sample_size: int,
) -> dict[str, Any]:
    risk = gauss_hermite_risk(
        posterior, features, base_scores, labels,
        config["selection_quadrature_order"], config["selection_chunk_size"],
    )
    raw, output = posterior.kl_values()
    bound = pac_bayes_certificate(risk, float(output), complexity_sample_size, delta)
    return {
        "candidate": candidate_name, "objective": objective_name,
        "learning_rate": learning_rate, "step": step,
        "selection_examples": labels.numel(),
        "B_gauss_hermite_risk": risk, "raw_kl": float(raw), "output_kl": float(output),
        "selection_upper": bound["population_gibbs_risk_upper"],
        "mean_l2": float(torch.linalg.vector_norm(posterior.mean.detach())),
        "std_min": float(posterior.std.detach().min()),
        "std_max": float(posterior.std.detach().max()),
    }


def _selection_key(summary: Mapping[str, Any]) -> tuple[float, float, str, int]:
    return summary["selection_upper"], summary["output_kl"], summary["candidate"], summary["step"]


def _sample_indices(
    number_examples: int, batch_size: int, generator: torch.Generator, device: torch.device
) -> Tensor:
    values = torch.randint(
        0, number_examples, (min(number_examples, batch_size),),
        generator=generator, dtype=torch.long,
    )
    return values.to(device)


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().to(torch.float64).clone() for name, value in state.items()}


# -----------------------------------------------------------------------------
# Fresh post-selection Monte Carlo and a conservative exact binomial endpoint.


@torch.inference_mode()
def fresh_monte_carlo(
    posterior: CanonicalPosterior,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    trials: int,
    chunk_size: int,
    seed: int,
    delta: float,
) -> dict[str, Any]:
    if features.device.type != "cpu" or features.dtype != torch.float64:
        raise ValueError("final Monte Carlo requires CPU float64 inputs")
    means, variances = posterior.score_statistics(features, base_scores)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    errors, completed = 0, 0
    while completed < trials:
        count = min(chunk_size, trials - completed)
        indices = torch.randint(0, labels.numel(), (count,), generator=generator)
        noise = torch.randn(count, posterior.number_classes, generator=generator, dtype=torch.float64)
        sampled_scores = means.index_select(0, indices) + torch.sqrt(
            variances.index_select(0, indices)
        ) * noise
        errors += int(torch.count_nonzero(sampled_scores.argmax(1) != labels.index_select(0, indices)))
        completed += count
    observed = errors / trials
    upper = clopper_pearson_upper(errors, trials, delta)
    if upper + 1e-15 < observed:
        raise RuntimeError("Clopper-Pearson endpoint fell below the observation")
    return {
        "seed": seed, "trials": trials, "errors": errors,
        "observed_risk": observed, "clopper_pearson_upper": upper,
    }


def clopper_pearson_upper(errors: int, trials: int, delta: float) -> float:
    error_count = _integer(errors, "errors")
    trial_count = _integer(trials, "trials")
    if trial_count <= 0 or error_count < 0 or error_count > trial_count:
        raise ValueError("invalid binomial counts")
    if error_count == trial_count:
        return 1.0
    if error_count == 0:
        endpoint = -math.expm1(math.log(delta) / trial_count)
    else:
        endpoint = float(betaincinv(error_count + 1, trial_count - error_count, 1.0 - delta))
    candidate = math.nextafter(endpoint, 1.0)
    delta_decimal = Decimal.from_float(float(delta))
    if _binomial_cdf_upper_bound(error_count, trial_count, candidate) <= delta_decimal:
        return candidate
    return _first_conservative_float(error_count, trial_count, delta_decimal, candidate)


def _first_conservative_float(errors: int, trials: int, delta: Decimal, candidate: float) -> float:
    low_bits, one_bits, stride = _float_bits(candidate), _float_bits(1.0), 1
    while True:
        high_bits = min(one_bits, low_bits + stride)
        if _binomial_cdf_upper_bound(errors, trials, _bits_float(high_bits)) <= delta:
            break
        low_bits, stride = high_bits, stride * 2
    while high_bits - low_bits > 1:
        midpoint = (low_bits + high_bits) // 2
        if _binomial_cdf_upper_bound(errors, trials, _bits_float(midpoint)) <= delta:
            high_bits = midpoint
        else:
            low_bits = midpoint
    return _bits_float(high_bits)


def _binomial_cdf_upper_bound(errors: int, trials: int, probability: float) -> Decimal:
    if probability <= 0.0:
        return Decimal(1)
    if probability >= 1.0:
        return Decimal(0) if errors < trials else Decimal(1)
    p = Decimal.from_float(probability)
    floor, ceiling = _decimal_context(ROUND_FLOOR), _decimal_context(ROUND_CEILING)
    one = Decimal(1)
    q_lower = floor.subtract(one, p)
    q_upper = ceiling.subtract(one, p)
    if errors + 1 <= trials - errors:
        term = _positive_integer_power(q_upper, trials, ceiling)
        total = term
        odds_upper = ceiling.divide(p, q_lower)
        for count in range(errors):
            ratio = ceiling.divide(Decimal(trials - count), Decimal(count + 1))
            term = ceiling.multiply(ceiling.multiply(term, ratio), odds_upper)
            total = ceiling.add(total, term)
        return min(one, total)
    term = _positive_integer_power(p, trials, floor)
    upper_tail_lower = term
    reverse_odds = floor.divide(q_lower, p)
    for count in range(trials, errors + 1, -1):
        ratio = floor.divide(Decimal(count), Decimal(trials - count + 1))
        term = floor.multiply(floor.multiply(term, ratio), reverse_odds)
        upper_tail_lower = floor.add(upper_tail_lower, term)
    return ceiling.subtract(one, max(Decimal(0), min(one, upper_tail_lower)))


def _positive_integer_power(value: Decimal, exponent: int, context: Context) -> Decimal:
    result, base, remaining = Decimal(1), value, exponent
    while remaining:
        if remaining & 1:
            result = context.multiply(result, base)
        remaining >>= 1
        if remaining:
            base = context.multiply(base, base)
    return result


def _decimal_context(rounding: str) -> Context:
    context = getcontext().copy()
    context.prec, context.rounding, context.Emin, context.Emax = 80, rounding, MIN_EMIN, MAX_EMAX
    return context


def _float_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _bits_float(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
