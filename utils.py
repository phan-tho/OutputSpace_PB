"""Small utilities shared by the experiment script."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import scipy
import torch


TOP_LEVEL_FIELDS = {
    "name",
    "dataset",
    "encoder",
    "prior",
    "feature_map",
    "posterior",
    "confidence",
    "certification",
    "numerics",
}

SECTION_FIELDS = {
    "dataset": {
        "name", "prior_fraction", "number_classes",
        "synthetic_train_size", "synthetic_test_size", "synthetic_input_dimension",
    },
    "encoder": {"name", "feature_dimension", "width", "dropout", "initialization_seed"},
    "prior": {
        "source", "optimizer", "epochs", "batch_size", "learning_rate",
        "weight_decay", "momentum", "warmup_epochs", "calibration_fraction",
        "label_smoothing", "augmentation", "cutout_size", "validation_every",
        "schedule", "lr_milestones", "lr_gamma",
    },
    "feature_map": {"kind", "rank", "kappa", "projection_seed", "normalization_epsilon"},
    "posterior": {
        "steps", "checkpoint_every", "batch_size", "learning_rates", "objectives",
        "warmup_steps", "training_quadrature_order", "selection_quadrature_order",
        "selection_chunk_size", "selection_examples", "gradient_clip_norm",
    },
    "confidence": {
        "total_failure_probability", "pac_bayes_delta_each",
        "pac_bayes_family_count", "monte_carlo_delta", "direct_holdout_delta",
    },
    "certification": {
        "monte_carlo_trials", "monte_carlo_chunk_size",
        "diagnostic_quadrature_order", "evaluate_test", "direct_holdout",
    },
    "numerics": {"minimum_posterior_std", "minimum_relative_singular_value"},
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - TOP_LEVEL_FIELDS
    missing = TOP_LEVEL_FIELDS - set(config)
    if unknown or missing:
        raise ValueError(f"configuration fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for section, allowed in SECTION_FIELDS.items():
        values = config[section]
        if not isinstance(values, Mapping):
            raise ValueError(f"config[{section!r}] must be an object")
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"unknown {section} fields: {sorted(extra)}")

    dataset = config["dataset"]
    encoder = config["encoder"]
    prior = config["prior"]
    feature_map = config["feature_map"]
    posterior = config["posterior"]
    confidence = config["confidence"]
    certification = config["certification"]
    numerics = config["numerics"]

    if dataset["name"] not in {"synthetic", "mnist", "cifar10", "cifar100", "imagenet"}:
        raise ValueError("unsupported dataset")
    if not 0.0 <= dataset["prior_fraction"] < 1.0 or dataset["number_classes"] < 2:
        raise ValueError("invalid dataset split fraction or class count")
    if encoder["feature_dimension"] <= 0:
        raise ValueError("encoder feature_dimension must be positive")
    if prior["source"] not in {"a_trained", "upstream", "random"}:
        raise ValueError("prior source must be a_trained, upstream, or random")
    if prior["source"] in {"upstream", "random"} and dataset["prior_fraction"] != 0.0:
        raise ValueError("upstream and random priors require an empty A block")
    if prior["source"] == "a_trained" and dataset["prior_fraction"] <= 0.0:
        raise ValueError("an A-trained prior requires a nonempty A block")
    if prior["optimizer"] not in {"adamw", "sgd"}:
        raise ValueError("prior optimizer must be adamw or sgd")
    if prior.get("schedule", "cosine") not in {"cosine", "step"}:
        raise ValueError("prior schedule must be cosine or step")
    if prior.get("schedule") == "step" and (
        not prior.get("lr_milestones") or not 0.0 < prior.get("lr_gamma", 0.0) < 1.0
    ):
        raise ValueError("step schedule needs milestones and lr_gamma in (0,1)")
    if prior["source"] == "a_trained" and min(prior["epochs"], prior["batch_size"]) <= 0:
        raise ValueError("prior epochs and batch size must be positive")
    if prior["source"] in {"upstream", "random"} and prior["epochs"] != 0:
        raise ValueError("upstream and random priors must use zero training epochs")
    if feature_map["kind"] not in {
        "standardize", "pca_whiten_bias", "upstream_pca", "random_projection_bias"
    }:
        raise ValueError("unknown feature map")
    if feature_map["rank"] <= 0:
        raise ValueError("feature rank must be positive")
    if feature_map["kind"] != "random_projection_bias" and feature_map["kappa"] <= 0.0:
        raise ValueError("feature kappa must be positive")
    if feature_map["kind"] == "standardize" and feature_map["rank"] != encoder["feature_dimension"]:
        raise ValueError("standardized feature rank must equal encoder dimension")
    if feature_map["kind"] == "pca_whiten_bias" and not 2 <= feature_map["rank"] <= encoder["feature_dimension"] + 1:
        raise ValueError("PCA rank must contain a bias and a valid number of components")
    if feature_map["kind"] == "upstream_pca" and prior["source"] != "upstream":
        raise ValueError("upstream_pca requires an upstream prior")
    if feature_map["kind"] == "random_projection_bias":
        if prior["source"] != "random":
            raise ValueError("random_projection_bias requires a random prior")
        if encoder["initialization_seed"] < 0:
            raise ValueError("random encoder initialization seed must be nonnegative")
        if not 2 <= feature_map["rank"] <= encoder["feature_dimension"] + 1:
            raise ValueError("random projection rank must contain a bias and fit the encoder")
        if feature_map["projection_seed"] < 0 or feature_map["normalization_epsilon"] <= 0.0:
            raise ValueError("random projection seed and epsilon must be positive")
    if min(
        posterior["steps"], posterior["checkpoint_every"], posterior["batch_size"],
        posterior["training_quadrature_order"], posterior["selection_quadrature_order"],
        posterior["selection_chunk_size"],
    ) <= 0:
        raise ValueError("posterior counts must be positive")
    if posterior["warmup_steps"] < 0 or not posterior["learning_rates"]:
        raise ValueError("invalid posterior warmup or learning-rate list")
    if posterior.get("selection_examples", 1) <= 0:
        raise ValueError("posterior selection_examples must be positive")
    if any(value <= 0.0 for value in posterior["learning_rates"]):
        raise ValueError("posterior learning rates must be positive")
    if not posterior["objectives"] or any(
        name not in {"classic", "quad", "exact"} for name in posterior["objectives"]
    ):
        raise ValueError("posterior objectives must be classic, quad, or exact")
    if len(set(posterior["objectives"])) != len(posterior["objectives"]):
        raise ValueError("posterior objectives must be unique")

    deltas = (
        confidence["total_failure_probability"],
        confidence["pac_bayes_delta_each"],
        confidence["monte_carlo_delta"],
        confidence["direct_holdout_delta"],
    )
    if any(not 0.0 < value < 1.0 for value in deltas):
        raise ValueError("confidence values must lie in (0,1)")
    family_count = confidence["pac_bayes_family_count"]
    allocated = confidence["pac_bayes_delta_each"] * family_count + confidence["monte_carlo_delta"]
    if family_count <= 0 or allocated > confidence["total_failure_probability"] + 1e-15:
        raise ValueError("PAC-Bayes family plus MC allocation exceeds total failure")
    if min(certification["monte_carlo_trials"], certification["monte_carlo_chunk_size"]) <= 0:
        raise ValueError("Monte Carlo counts must be positive")
    if not 0.0 < numerics["minimum_posterior_std"] < 1.0:
        raise ValueError("minimum posterior standard deviation must lie in (0,1)")
    if not 0.0 <= numerics["minimum_relative_singular_value"] < 1.0:
        raise ValueError("invalid relative singular-value audit threshold")


def smoke_config() -> dict[str, Any]:
    config = {
        "name": "synthetic-smoke",
        "dataset": {
            "name": "synthetic", "prior_fraction": 0.75,
            "number_classes": 4, "synthetic_train_size": 128,
            "synthetic_test_size": 64, "synthetic_input_dimension": 12,
        },
        "encoder": {"name": "mlp", "feature_dimension": 8, "width": 16, "dropout": 0.0},
        "prior": {
            "source": "a_trained", "optimizer": "adamw", "epochs": 2,
            "batch_size": 32, "learning_rate": 0.01, "weight_decay": 0.0001,
            "momentum": 0.9, "warmup_epochs": 0, "calibration_fraction": 0.0,
            "label_smoothing": 0.0, "augmentation": "none", "cutout_size": 0,
            "validation_every": 1,
        },
        "feature_map": {"kind": "standardize", "rank": 8, "kappa": 0.1},
        "posterior": {
            "steps": 3, "checkpoint_every": 1, "batch_size": 32,
            "learning_rates": [0.01], "objectives": ["exact"], "warmup_steps": 1,
            "training_quadrature_order": 8, "selection_quadrature_order": 12,
            "selection_chunk_size": 32, "gradient_clip_norm": 10.0,
        },
        "confidence": {
            "total_failure_probability": 0.05, "pac_bayes_delta_each": 0.045,
            "pac_bayes_family_count": 1, "monte_carlo_delta": 0.005,
            "direct_holdout_delta": 0.05,
        },
        "certification": {
            "monte_carlo_trials": 4000, "monte_carlo_chunk_size": 1000,
            "diagnostic_quadrature_order": 16,
            "evaluate_test": True, "direct_holdout": True,
        },
        "numerics": {
            "minimum_posterior_std": 0.0001,
            "minimum_relative_singular_value": 1e-12,
        },
    }
    validate_config(config)
    return config


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def runtime_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "cuda_device_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if device.type == "cuda" else []
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")
