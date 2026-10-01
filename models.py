"""Backbones, the deterministic prior, feature maps, and canonical posterior."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import IndexedDataset, split_A_for_checkpoint_selection, upstream_input_transform


# -----------------------------------------------------------------------------
# Backbones. Each one exposes feature_dim and returns one feature vector/image.


class MlpBackbone(nn.Module):
    def __init__(self, input_dimension: int, feature_dimension: int, width: int) -> None:
        super().__init__()
        self.feature_dim = feature_dimension
        self.network = nn.Sequential(
            nn.Linear(input_dimension, width), nn.ReLU(),
            nn.Linear(width, feature_dimension), nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class MnistCnnBackbone(nn.Module):
    def __init__(self, feature_dimension: int, width: int) -> None:
        super().__init__()
        self.feature_dim = feature_dimension
        self.network = nn.Sequential(
            nn.Conv2d(1, width, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(2 * width * 7 * 7, feature_dimension), nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class LeNet5Backbone(nn.Module):
    feature_dim = 84

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, self.feature_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        values = F.max_pool2d(F.relu(self.conv1(inputs)), 2)
        values = F.max_pool2d(F.relu(self.conv2(values)), 2)
        values = F.relu(self.fc1(values.flatten(1)))
        return F.relu(self.fc2(values))


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = dropout
        self.shortcut = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        values = F.relu(self.bn1(inputs), inplace=False)
        shortcut_input = inputs if isinstance(self.shortcut, nn.Identity) else values
        shortcut = self.shortcut(shortcut_input)
        values = self.conv1(values)
        values = F.relu(self.bn2(values), inplace=False)
        if self.dropout:
            values = F.dropout(values, p=self.dropout, training=self.training)
        return shortcut + self.conv2(values)


class WideResNet28x4Backbone(nn.Module):
    def __init__(self, feature_dimension: int, dropout: float) -> None:
        super().__init__()
        self.feature_dim = feature_dimension
        channels = (16, 64, 128, 256)
        self.stem = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.group1 = self._group(channels[0], channels[1], 1, dropout)
        self.group2 = self._group(channels[1], channels[2], 2, dropout)
        self.group3 = self._group(channels[2], channels[3], 2, dropout)
        self.final_bn = nn.BatchNorm2d(channels[3])
        self.projection = nn.Linear(channels[3], feature_dimension)
        self.layer_norm = nn.LayerNorm(feature_dimension)
        self._initialize()

    @staticmethod
    def _group(in_channels: int, out_channels: int, stride: int, dropout: float) -> nn.Sequential:
        blocks = [WideBasicBlock(in_channels, out_channels, stride, dropout)]
        blocks.extend(WideBasicBlock(out_channels, out_channels, 1, dropout) for _ in range(3))
        return nn.Sequential(*blocks)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.group3(self.group2(self.group1(self.stem(inputs))))
        values = F.relu(self.final_bn(values), inplace=False)
        values = F.adaptive_avg_pool2d(values, 1).flatten(1)
        return torch.tanh(self.layer_norm(self.projection(values)))


class ImageNetResNet18Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, weights_path: Path | None) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        if weights_path is None:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        else:
            model = resnet18(weights=None)
            payload = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(payload, Mapping) and "state_dict" in payload:
                payload = payload["state_dict"]
            if not isinstance(payload, Mapping):
                raise ValueError("ResNet checkpoint must contain a state dictionary")
            state = {
                str(key).removeprefix("module."): value
                for key, value in payload.items() if isinstance(value, Tensor)
            }
            model.load_state_dict(state, strict=True)
        model.fc = nn.Identity()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class ResNet18Backbone(nn.Module):
    """Randomly initialized torchvision ResNet-18 for an A-trained prior."""

    feature_dim = 512

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet18

        self.model = resnet18(weights=None)
        self.model.fc = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


BACKBONES = {
    "mlp": MlpBackbone,
    "mnist_cnn": MnistCnnBackbone,
    "lenet5": LeNet5Backbone,
    "wrn28_4": WideResNet28x4Backbone,
    "resnet18": ResNet18Backbone,
    "imagenet_resnet18": ImageNetResNet18Backbone,
}


def make_backbone(
    config: Mapping[str, Any], input_dimension: int | None, weights_path: Path | None
) -> nn.Module:
    name = config["name"]
    if name == "mlp":
        if input_dimension is None:
            raise ValueError("the MLP needs an input dimension")
        backbone = BACKBONES[name](input_dimension, config["feature_dimension"], config["width"])
    elif name == "mnist_cnn":
        backbone = BACKBONES[name](config["feature_dimension"], config["width"])
    elif name == "lenet5":
        backbone = BACKBONES[name]()
    elif name == "wrn28_4":
        backbone = BACKBONES[name](config["feature_dimension"], config["dropout"])
    elif name == "resnet18":
        if weights_path is not None:
            raise ValueError("the A-trained ResNet-18 must start from its fixed random initialization")
        backbone = BACKBONES[name]()
    elif name == "imagenet_resnet18":
        backbone = BACKBONES[name](weights_path)
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    if backbone.feature_dim != config["feature_dimension"]:
        raise ValueError("backbone feature dimension differs from the configuration")
    return backbone


# -----------------------------------------------------------------------------
# Deterministic prior. All fitting and checkpoint selection receives A indices.


class PriorModel(nn.Module):
    def __init__(self, backbone: nn.Module, number_classes: int, zero_head: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.number_classes = number_classes
        self.head = None if zero_head else nn.Linear(backbone.feature_dim, number_classes)

    def raw_features(self, inputs: Tensor) -> Tensor:
        return self.backbone(inputs)

    def base_scores_from_features(self, features: Tensor) -> Tensor:
        if self.head is None:
            return features.new_zeros((features.shape[0], self.number_classes))
        return self.head(features)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.raw_features(inputs)
        return self.base_scores_from_features(features)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def train_or_load_prior(
    model: PriorModel,
    images: Any,
    labels: Tensor,
    A_indices: Tensor,
    input_transform: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    workers: int,
    seed: int,
    checkpoint: Path | None,
) -> dict[str, Any]:
    model.to(device)
    if config["source"] in {"upstream", "random"}:
        model.freeze()
        return {
            "selected_epoch": 0, "selected_calibration_error": None,
            "final_training_loss": None, "checkpoint_loaded": False,
            "training_gpu_count": 0,
        }

    if A_indices.numel() == 0:
        raise ValueError("A-trained prior received an empty A block")
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise ValueError("prior checkpoint must contain state_dict and metadata")
        metadata = payload.get("metadata", {})
        if metadata.get("number_classes") != model.number_classes:
            raise RuntimeError("prior checkpoint class count does not match")
        saved_A_indices = metadata.get("A_indices")
        if not isinstance(saved_A_indices, Tensor) or not torch.equal(saved_A_indices, A_indices):
            raise RuntimeError("prior checkpoint was not trained on this A split")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.freeze()
        return {
            "selected_epoch": int(metadata.get("selected_epoch", 0)),
            "selected_calibration_error": None, "final_training_loss": None,
            "checkpoint_loaded": True, "training_gpu_count": 0,
        }

    fit_indices, calibration_indices = split_A_for_checkpoint_selection(
        A_indices, config["calibration_fraction"], seed + 1
    )
    training_set = IndexedDataset(
        images, labels, fit_indices, input_transform,
        cifar_augmentation=config["augmentation"] == "cifar",
        imagenet_augmentation=config["augmentation"] == "imagenet",
        cutout_size=config["cutout_size"],
    )
    distributed = dist.is_available() and dist.is_initialized()
    sampler = DistributedSampler(
        training_set, shuffle=True, seed=seed + 2
    ) if distributed else None
    loader = DataLoader(
        training_set, batch_size=config["batch_size"], shuffle=sampler is None,
        sampler=sampler, generator=torch.Generator().manual_seed(
            seed + 2 + (dist.get_rank() if distributed else 0)
        ), num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    training_gpu_count = dist.get_world_size() if distributed else int(device.type == "cuda")
    training_model: nn.Module = model
    if distributed:
        training_model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=1e-5
        )
    else:
        optimizer = torch.optim.SGD(
            parameters, lr=config["learning_rate"], momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"], nesterov=True,
        )
        scheduler = None

    best_state, best_epoch = None, 0
    best_calibration = (math.inf, math.inf)
    final_training_loss = math.nan
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(1, config["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if config["optimizer"] == "sgd":
            if config.get("schedule", "cosine") == "step":
                decays = sum(epoch > milestone for milestone in config["lr_milestones"])
                factor = config["lr_gamma"] ** decays
            else:
                warmup = config.get("warmup_epochs", 0)
                if warmup and epoch <= warmup:
                    factor = epoch / warmup
                else:
                    progress = (epoch - warmup) / max(1, config["epochs"] - warmup)
                    factor = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * factor

        training_model.train()
        loss_sum, examples_seen = 0.0, 0
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = F.cross_entropy(
                    training_model(batch_images), batch_labels,
                    label_smoothing=config["label_smoothing"],
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * batch_labels.numel()
            examples_seen += batch_labels.numel()
        if distributed:
            totals = torch.tensor([loss_sum, examples_seen], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            loss_sum, examples_seen = float(totals[0]), int(totals[1])
        final_training_loss = loss_sum / examples_seen
        if not distributed or dist.get_rank() == 0:
            print(
                f"[prior] epoch={epoch}/{config['epochs']} "
                f"loss={final_training_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

        validation_every = config.get("validation_every", 1)
        if calibration_indices.numel() and (epoch % validation_every == 0 or epoch == config["epochs"]):
            calibration_loss, calibration_error = prior_loss_and_error(
                model, images, labels, calibration_indices, input_transform,
                max(config["batch_size"], 256), device, workers,
            )
            if (calibration_loss, calibration_error) < best_calibration:
                best_calibration = (calibration_loss, calibration_error)
                best_epoch = epoch
                best_state = _cpu_state(model.state_dict())
        elif not calibration_indices.numel() and epoch == config["epochs"]:
            best_epoch = epoch
            best_state = _cpu_state(model.state_dict())
        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("prior training selected no model state")
    model.load_state_dict(best_state, strict=True)
    model.freeze()
    return {
        "selected_epoch": best_epoch,
        "selected_calibration_error": (
            None if not math.isfinite(best_calibration[1]) else best_calibration[1]
        ),
        "final_training_loss": final_training_loss,
        "checkpoint_loaded": False,
        "training_gpu_count": training_gpu_count,
    }


@torch.inference_mode()
def extract_prior_outputs(
    model: PriorModel,
    images: Any,
    labels: Tensor,
    indices: Tensor,
    input_transform: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    workers: int,
    *,
    include_scores: bool = True,
    output_dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor]:
    loader = DataLoader(
        IndexedDataset(images, labels, indices, input_transform), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model.eval()
    feature_parts, score_parts, label_parts = [], [], []
    for batch_images, batch_labels in loader:
        raw_features = model.raw_features(batch_images.to(device, non_blocking=True))
        feature_parts.append(raw_features.cpu().to(output_dtype))
        if include_scores:
            base_scores = model.base_scores_from_features(raw_features)
            score_parts.append(base_scores.cpu().to(output_dtype))
        label_parts.append(batch_labels.to(torch.long))
    features = torch.cat(feature_parts)
    scores = (
        torch.cat(score_parts) if include_scores
        else torch.empty((features.shape[0], 0), dtype=output_dtype)
    )
    return features, scores, torch.cat(label_parts)


def prior_loss_and_error(
    model: PriorModel,
    images: Any,
    labels: Tensor,
    indices: Tensor,
    input_transform: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    workers: int,
) -> tuple[float, float]:
    _, scores, selected_labels = extract_prior_outputs(
        model, images, labels, indices, input_transform, batch_size, device, workers
    )
    return (
        float(F.cross_entropy(scores, selected_labels)),
        float((scores.argmax(1) != selected_labels).to(torch.float64).mean()),
    )


# -----------------------------------------------------------------------------
# A-only (or externally fixed upstream) feature transformation.


def fit_feature_transform(raw_A: Tensor, config: Mapping[str, Any]) -> dict[str, Any]:
    values = raw_A.detach().cpu().to(torch.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not bool(torch.all(torch.isfinite(values))):
        raise ValueError("feature fitting needs at least two finite A feature vectors")
    if config["kind"] == "standardize":
        mean = values.mean(0)
        std = values.std(0, unbiased=False)
        if bool(torch.any(std <= 0.0)):
            raise RuntimeError("A-only standardization found a zero direction")
        return {"kind": "standardize", "mean": mean, "std": std.clamp_min(1e-4),
                "kappa": config["kappa"]}
    if config["kind"] == "pca_whiten_bias":
        mean = values.mean(0)
        covariance = (values - mean).T @ (values - mean) / values.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[: config["rank"] - 1]
        retained = eigenvalues.index_select(0, order)
        if float(retained[-1]) <= 0.0 or not bool(torch.all(torch.isfinite(retained))):
            raise RuntimeError("requested A-only PCA rank contains a zero direction")
        directions = eigenvectors.index_select(1, order)
        pivots = directions.abs().argmax(0)
        signs = torch.sign(directions[pivots, torch.arange(directions.shape[1])])
        signs[signs == 0] = 1
        return {
            "kind": "pca_whiten_bias", "mean": mean,
            "directions": directions * signs.unsqueeze(0),
            "inverse_scales": torch.rsqrt(retained), "eigenvalues": retained,
            "kappa": config["kappa"],
        }
    raise ValueError("upstream feature transforms must be loaded from an artifact")


def make_random_feature_transform(
    raw_dimension: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Fixed random projection chosen before any downstream image is read."""

    components = config["rank"] - 1
    if not 1 <= components <= raw_dimension:
        raise ValueError("random projection rank exceeds the encoder feature dimension")
    generator = torch.Generator().manual_seed(config["projection_seed"])
    draw = torch.randn(raw_dimension, components, generator=generator, dtype=torch.float64)
    basis, triangular = torch.linalg.qr(draw, mode="reduced")
    signs = torch.sign(torch.diagonal(triangular))
    signs[signs == 0.0] = 1.0
    directions = (basis * signs).T.contiguous()
    return {
        "kind": "random_projection_bias",
        "directions": directions,
        "normalization_epsilon": config["normalization_epsilon"],
        "projection_seed": config["projection_seed"],
    }


def apply_feature_transform(raw_features: Tensor, transform: Mapping[str, Any]) -> Tensor:
    values = raw_features.detach().cpu().to(torch.float64)
    if transform["kind"] == "standardize":
        rank = transform["mean"].numel()
        standardized = (values - transform["mean"]) / transform["std"]
        return (transform["kappa"] / math.sqrt(rank) * standardized).contiguous()
    if transform["kind"] == "random_projection_bias":
        normalized = math.sqrt(values.shape[1]) * values / torch.sqrt(
            values.square().sum(1, keepdim=True) + transform["normalization_epsilon"]
        )
        projected = normalized @ transform["directions"].T
        bias = torch.ones((values.shape[0], 1), dtype=torch.float64)
        return (torch.cat((bias, projected), 1) / math.sqrt(projected.shape[1] + 1)).contiguous()
    whitened = ((values - transform["mean"]) @ transform["directions"]) * transform["inverse_scales"]
    bias = torch.ones((values.shape[0], 1), dtype=torch.float64)
    output_dimension = transform["directions"].shape[1] + 1
    result = transform["kappa"] * torch.cat((bias, whitened), 1) / math.sqrt(output_dimension)
    if not bool(torch.all(torch.isfinite(result))):
        raise FloatingPointError("feature transform produced non-finite values")
    return result.contiguous()


def feature_transform_report(transform: Mapping[str, Any]) -> dict[str, Any]:
    if transform["kind"] == "standardize":
        return {
            "kind": "standardize", "output_dimension": int(transform["mean"].numel()),
            "kappa": transform["kappa"], "minimum_A_standard_deviation": float(transform["std"].min()),
        }
    if transform["kind"] == "random_projection_bias":
        return {
            "kind": transform["kind"],
            "output_dimension": int(transform["directions"].shape[0] + 1),
            "projection_seed": transform["projection_seed"],
            "normalization_epsilon": transform["normalization_epsilon"],
        }
    report = {
        "kind": transform["kind"],
        "output_dimension": int(transform["directions"].shape[1] + 1),
        "pca_components": int(transform["directions"].shape[1]), "kappa": transform["kappa"],
    }
    if "eigenvalues" in transform:
        report["smallest_retained_A_eigenvalue"] = float(transform["eigenvalues"][-1])
    return report


def load_upstream_feature_transform(
    path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("upstream artifact must be a mapping")
    required = {"artifact_type", "backbone", "feature_dimension", "labels_read",
                "preprocessing", "tensors"}
    if not required.issubset(payload):
        raise ValueError("upstream artifact is missing required fields")
    if payload["artifact_type"] != "imagenet_penultimate_pca_whitening":
        raise ValueError("unexpected upstream artifact type")
    if payload["backbone"] != "resnet18" or int(payload["feature_dimension"]) != 512:
        raise ValueError("upstream artifact is not for ResNet-18")
    if bool(payload["labels_read"]):
        raise RuntimeError("upstream statistics must not have read labels")
    tensors = payload["tensors"]
    for name in ("feature_mean", "top_eigenvectors", "top_inverse_sqrt_eigenvalues"):
        if not isinstance(tensors.get(name), Tensor):
            raise ValueError(f"upstream tensor {name} is missing")
    components = config["rank"] - 1
    directions = tensors["top_eigenvectors"][:, :components].to(torch.float64)
    inverse_scales = tensors["top_inverse_sqrt_eigenvalues"][:components].to(torch.float64)
    if directions.shape != (512, components) or inverse_scales.shape != (components,):
        raise ValueError("upstream artifact lacks the requested PCA width")
    transform = {
        "kind": "upstream_pca", "mean": tensors["feature_mean"].to(torch.float64),
        "directions": directions, "inverse_scales": inverse_scales,
        "kappa": config["kappa"],
    }
    audit = {
        "image_count": int(payload.get("image_count", 0)), "labels_read": False,
    }
    return transform, upstream_input_transform(payload["preprocessing"]), audit


def validate_upstream_backbone(backbone: nn.Module) -> None:
    if not isinstance(backbone, ImageNetResNet18Backbone):
        raise ValueError("upstream PCA requires the ImageNet ResNet-18 backbone")


# -----------------------------------------------------------------------------
# The only stochastic method: symmetric independent class-score canonical lift.


def raw_kl(means: Tensor, stds: Tensor) -> Tensor:
    log_variances = 2.0 * torch.log(stds)
    return 0.5 * torch.sum(torch.expm1(log_variances) - log_variances + means.square())


def quotient_kl(means: Tensor, stds: Tensor) -> Tensor:
    classes, rank = means.shape
    centered_means = means - means.mean(0, keepdim=True)
    log_variances = 2.0 * torch.log(stds)
    covariance_term = (
        ((classes - 1.0) / classes) * torch.expm1(log_variances).sum(0)
        - log_variances.sum(0)
        - (torch.logsumexp(-log_variances, dim=0) - means.new_tensor(float(classes)).log())
    )
    result = 0.5 * (centered_means.square().sum(0) + covariance_term).sum()
    tolerance = 1000.0 * torch.finfo(result.dtype).eps * max(1, (classes - 1) * rank)
    if float(result.detach()) < -tolerance:
        raise FloatingPointError("quotient KL was materially negative")
    return result.clamp_min(0.0)


class CanonicalPosterior(nn.Module):
    def __init__(self, right_basis: Tensor, number_classes: int, minimum_std: float) -> None:
        super().__init__()
        if right_basis.ndim != 2 or min(right_basis.shape) <= 0:
            raise ValueError("right_basis must have shape [feature_dimension, observed_rank]")
        identity = torch.eye(
            right_basis.shape[1], dtype=right_basis.dtype, device=right_basis.device
        )
        tolerance = 2e-6 if right_basis.dtype == torch.float32 else 2e-12
        if not torch.allclose(right_basis.T @ right_basis, identity, atol=tolerance, rtol=tolerance):
            raise ValueError("right_basis columns must be orthonormal")
        self.number_classes = number_classes
        self.minimum_std = minimum_std
        self.register_buffer("right_basis", right_basis.detach().clone())
        shape = (number_classes, right_basis.shape[1])
        self.mean = nn.Parameter(torch.zeros(shape, dtype=right_basis.dtype, device=right_basis.device))
        initial = right_basis.new_tensor(1.0 - minimum_std)
        self.raw_std = nn.Parameter(torch.log(torch.expm1(initial)).expand(shape).clone())

    @property
    def std(self) -> Tensor:
        return self.minimum_std + F.softplus(self.raw_std)

    def score_statistics(self, features: Tensor, base_scores: Tensor) -> tuple[Tensor, Tensor]:
        projected = features @ self.right_basis
        means = base_scores + projected @ self.mean.T
        visible_variance = projected.square() @ self.std.square().T
        residual = features - projected @ self.right_basis.T
        variances = visible_variance + residual.square().sum(1, keepdim=True)
        if bool(torch.any(variances < 0.0)):
            raise FloatingPointError("score variance became negative")
        return means, variances

    def kl_values(self) -> tuple[Tensor, Tensor]:
        raw = raw_kl(self.mean, self.std)
        output = quotient_kl(self.mean, self.std)
        tolerance = 1000.0 * torch.finfo(raw.dtype).eps * max(1, self.mean.numel())
        if float(output.detach() - raw.detach()) > tolerance:
            raise FloatingPointError("quotient KL exceeds raw KL for the same posterior")
        return raw, output

    @torch.no_grad()
    def remove_common_mean(self) -> None:
        self.mean.sub_(self.mean.mean(0, keepdim=True))

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}
