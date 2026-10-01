"""Data loading, index-only splitting, and input preprocessing."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_training_set(
    config: Mapping[str, Any], data_root: Path, download: bool, experiment_seed: int
) -> tuple[Any, Tensor]:
    return _load_set(config, data_root, train=True, download=download, seed=experiment_seed)


def load_test_set(
    config: Mapping[str, Any], data_root: Path, download: bool, experiment_seed: int
) -> tuple[Any, Tensor]:
    """Called only after the certificate is frozen."""

    return _load_set(config, data_root, train=False, download=download, seed=experiment_seed)


def observation_independent_split(
    number_examples: int, prior_fraction: float, split_seed: int
) -> tuple[Tensor, Tensor]:
    """The function cannot inspect observations because it accepts only n and a seed."""

    if prior_fraction == 0.0:
        A_indices = torch.empty(0, dtype=torch.long)
        B_indices = torch.arange(number_examples, dtype=torch.long)
    else:
        permutation = torch.randperm(
            number_examples, generator=torch.Generator().manual_seed(split_seed)
        )
        number_A = int(round(prior_fraction * number_examples))
        if not 0 < number_A < number_examples:
            raise ValueError("A/B split produced an empty block")
        A_indices = permutation[:number_A]
        B_indices = permutation[number_A:]

    all_indices = torch.cat((A_indices, B_indices))
    if all_indices.numel() != number_examples or torch.unique(all_indices).numel() != number_examples:
        raise RuntimeError("A/B split does not form a disjoint partition")
    return A_indices, B_indices


def split_A_for_checkpoint_selection(
    A_indices: Tensor, calibration_fraction: float, seed: int
) -> tuple[Tensor, Tensor]:
    if calibration_fraction == 0.0:
        return A_indices.clone(), torch.empty(0, dtype=torch.long)
    order = torch.randperm(A_indices.numel(), generator=torch.Generator().manual_seed(seed))
    number_calibration = max(1, int(round(calibration_fraction * A_indices.numel())))
    return (
        A_indices.index_select(0, order[number_calibration:]),
        A_indices.index_select(0, order[:number_calibration]),
    )


def fit_A_only_input_transform(
    images: Any, A_indices: Tensor, dataset_name: str
) -> dict[str, Any]:
    if dataset_name == "synthetic":
        return {"kind": "identity"}
    if dataset_name == "imagenet":
        # Fixed public preprocessing; it is independent of both A and B.
        return {
            "kind": "imagenet_standard",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize_size": 256,
            "crop_size": 224,
        }
    selected = images.index_select(0, A_indices).to(torch.float64).div(255.0)
    mean = selected.mean(dim=(0, 2, 3))
    std = selected.std(dim=(0, 2, 3), unbiased=False)
    if bool(torch.any(std <= 0.0)):
        raise RuntimeError("A-only input normalization found a zero-variance channel")
    return {
        "kind": "channel_standardize",
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }


def upstream_input_transform(preprocessing: Mapping[str, Any]) -> dict[str, Any]:
    if "BILINEAR" not in str(preprocessing["interpolation"]):
        raise ValueError("only the audited bilinear ImageNet preprocessing is supported")
    if not bool(preprocessing["antialias"]):
        raise ValueError("upstream preprocessing must use antialiasing")

    def scalar_size(value: Any) -> int:
        return int(value[0]) if isinstance(value, (list, tuple)) else int(value)

    return {
        "kind": "imagenet_v1",
        "mean": [float(value) for value in preprocessing["mean"]],
        "std": [float(value) for value in preprocessing["std"]],
        "resize_size": scalar_size(preprocessing["resize_size"]),
        "crop_size": scalar_size(preprocessing["crop_size"]),
    }


class IndexedDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        images: Any,
        labels: Tensor,
        indices: Tensor,
        input_transform: Mapping[str, Any],
        *,
        cifar_augmentation: bool = False,
        imagenet_augmentation: bool = False,
        cutout_size: int = 0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.input_transform = input_transform
        self.cifar_augmentation = cifar_augmentation
        self.imagenet_augmentation = imagenet_augmentation
        self.cutout_size = cutout_size

    def __len__(self) -> int:
        return self.indices.numel()

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor]:
        index = int(self.indices[position])
        image = self.images[index]
        if self.cifar_augmentation:
            image = _augment_cifar(image, self.input_transform, self.cutout_size)
        elif self.imagenet_augmentation:
            image = _augment_imagenet(image, self.input_transform)
        else:
            image = transform_image(image, self.input_transform)
        return image, self.labels[index]


def transform_image(image: Tensor, transform: Mapping[str, Any]) -> Tensor:
    kind = transform["kind"]
    if kind == "identity":
        return image.to(torch.float32)
    if kind == "imagenet_standard":
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        value = TF.resize(
            image, [transform["resize_size"]],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        value = TF.center_crop(value, [transform["crop_size"]])
        value = value.to(torch.float32).div(255.0)
        mean = value.new_tensor(transform["mean"])[:, None, None]
        std = value.new_tensor(transform["std"])[:, None, None]
        return (value - mean) / std
    value = image.to(torch.float32).div(255.0)
    if kind == "fixed_minus_one_one":
        return value.mul(2.0).sub(1.0)
    if kind == "channel_standardize":
        mean = value.new_tensor(transform["mean"])[:, None, None]
        std = value.new_tensor(transform["std"])[:, None, None]
        return (value - mean) / std
    if kind == "imagenet_v1":
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        pil_image = Image.fromarray(image.permute(1, 2, 0).contiguous().numpy(), mode="RGB")
        resized = TF.resize(
            pil_image,
            [transform["resize_size"]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        cropped = TF.center_crop(resized, [transform["crop_size"]])
        value = TF.pil_to_tensor(cropped).to(torch.float32).div(255.0)
        mean = value.new_tensor(transform["mean"])[:, None, None]
        std = value.new_tensor(transform["std"])[:, None, None]
        return (value - mean) / std
    raise ValueError(f"unknown input transform: {kind}")


def _augment_cifar(
    image: Tensor, input_transform: Mapping[str, Any], cutout_size: int
) -> Tensor:
    value = image.to(torch.float32).div(255.0)
    padded = F.pad(value, (4, 4, 4, 4), mode="reflect")
    top = int(torch.randint(0, 9, ()).item())
    left = int(torch.randint(0, 9, ()).item())
    value = padded[:, top : top + 32, left : left + 32]
    if bool(torch.rand(()) < 0.5):
        value = torch.flip(value, dims=(2,))
    mean = value.new_tensor(input_transform["mean"])[:, None, None]
    std = value.new_tensor(input_transform["std"])[:, None, None]
    value = (value - mean) / std
    if cutout_size:
        center_y = int(torch.randint(0, 32, ()).item())
        center_x = int(torch.randint(0, 32, ()).item())
        half = cutout_size // 2
        value[
            :,
            max(0, center_y - half) : min(32, center_y + half),
            max(0, center_x - half) : min(32, center_x + half),
        ] = 0.0
    return value


def _augment_imagenet(image: Tensor, transform: Mapping[str, Any]) -> Tensor:
    from torchvision.transforms import InterpolationMode, RandomResizedCrop
    from torchvision.transforms import functional as TF

    top, left, height, width = RandomResizedCrop.get_params(
        image, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
    )
    value = TF.resized_crop(
        image, top, left, height, width,
        [transform["crop_size"], transform["crop_size"]],
        interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    if bool(torch.rand(()) < 0.5):
        value = torch.flip(value, dims=(2,))
    value = value.to(torch.float32).div(255.0)
    mean = value.new_tensor(transform["mean"])[:, None, None]
    std = value.new_tensor(transform["std"])[:, None, None]
    return (value - mean) / std


def _load_set(
    config: Mapping[str, Any],
    data_root: Path,
    *,
    train: bool,
    download: bool,
    seed: int,
) -> tuple[Any, Tensor]:
    name = config["name"]
    if name == "synthetic":
        count = config["synthetic_train_size"] if train else config["synthetic_test_size"]
        data_generator = torch.Generator().manual_seed(seed + (0 if train else 10_000))
        inputs = torch.randn(count, config["synthetic_input_dimension"], generator=data_generator)
        weight_generator = torch.Generator().manual_seed(seed + 99)
        weights = torch.randn(
            config["synthetic_input_dimension"], config["number_classes"], generator=weight_generator
        )
        labels = (
            inputs @ weights
            + 0.15 * torch.randn(count, config["number_classes"], generator=data_generator)
        ).argmax(dim=1)
        return inputs, labels.to(torch.long)
    if name == "mnist":
        from torchvision.datasets import MNIST

        root = data_root.parent if (data_root / "raw").is_dir() else data_root
        dataset = MNIST(root=str(root), train=train, download=download)
        return dataset.data.unsqueeze(1).contiguous(), dataset.targets.to(torch.long)
    if name in {"cifar10", "cifar100"}:
        directory = _find_cifar_python_directory(name, data_root)
        if directory is not None:
            return _load_cifar_python(name, directory, train)
        if not download:
            raise FileNotFoundError(f"no {name} Python-batch directory below {data_root}")
        from torchvision.datasets import CIFAR10, CIFAR100

        dataset_class = CIFAR10 if name == "cifar10" else CIFAR100
        dataset = dataset_class(root=str(data_root), train=train, download=True)
        images = torch.from_numpy(np.asarray(dataset.data).copy()).permute(0, 3, 1, 2)
        return images.contiguous(), torch.tensor(dataset.targets, dtype=torch.long)
    if name == "imagenet":
        if download:
            raise ValueError("ImageNet must be downloaded separately; do not use --download")
        return _load_imagenet_paths(data_root, train)
    raise ValueError(f"unsupported dataset: {name}")


class ImageNetPaths:
    """A lightweight path collection; images are decoded only when requested."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        from torchvision.io import ImageReadMode, read_image

        return read_image(str(self.paths[index]), mode=ImageReadMode.RGB)


def _load_imagenet_paths(data_root: Path, train: bool) -> tuple[ImageNetPaths, Tensor]:
    candidates = [
        data_root / "ILSVRC" / "Data" / "CLS-LOC",
        data_root / "Data" / "CLS-LOC",
        data_root,
    ]
    cls_loc = next(
        (path for path in candidates if (path / "train").is_dir() and (path / "val").is_dir()),
        None,
    )
    if cls_loc is None:
        raise FileNotFoundError(
            f"expected ILSVRC/Data/CLS-LOC/{{train,val}} below {data_root}"
        )

    class_directories = sorted(path for path in (cls_loc / "train").iterdir() if path.is_dir())
    if len(class_directories) != 1000:
        raise RuntimeError(f"expected 1000 ImageNet classes, found {len(class_directories)}")
    class_index = {path.name: index for index, path in enumerate(class_directories)}

    if train:
        paths, labels = [], []
        for synset_directory in class_directories:
            class_paths = sorted(synset_directory.glob("*.JPEG"))
            paths.extend(class_paths)
            labels.extend([class_index[synset_directory.name]] * len(class_paths))
        if len(paths) != 1_281_167:
            raise RuntimeError(f"expected 1,281,167 ImageNet training images, found {len(paths)}")
        return ImageNetPaths(paths), torch.tensor(labels, dtype=torch.long)

    solution_candidates = [
        data_root / "LOC_val_solution.csv",
        cls_loc.parents[2] / "LOC_val_solution.csv",
    ]
    solution_path = next((path for path in solution_candidates if path.is_file()), None)
    if solution_path is None:
        raise FileNotFoundError("LOC_val_solution.csv is required for diagnostic validation labels")
    labels_by_id = {}
    with solution_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels_by_id[row["ImageId"]] = class_index[row["PredictionString"].split()[0]]
    paths = sorted((cls_loc / "val").glob("*.JPEG"))
    labels = torch.tensor([labels_by_id[path.stem] for path in paths], dtype=torch.long)
    if len(paths) != 50_000:
        raise RuntimeError(f"expected 50,000 ImageNet validation images, found {len(paths)}")
    return ImageNetPaths(paths), labels


def _find_cifar_python_directory(name: str, root: Path) -> Path | None:
    required = (
        [f"data_batch_{index}" for index in range(1, 6)] + ["test_batch"]
        if name == "cifar10" else ["train", "test", "meta"]
    )
    matches = []
    directories = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for directory in directories:
        if all((directory / filename).is_file() for filename in required):
            matches.append(directory.resolve())
    matches = sorted(set(matches))
    if len(matches) > 1:
        raise RuntimeError(f"found multiple {name} Python-batch directories")
    return matches[0] if matches else None


def _load_cifar_python(name: str, directory: Path, train: bool) -> tuple[Tensor, Tensor]:
    if name == "cifar10":
        filenames = [f"data_batch_{index}" for index in range(1, 6)] if train else ["test_batch"]
        label_name, expected_classes = "labels", 10
    else:
        filenames = ["train"] if train else ["test"]
        label_name, expected_classes = "fine_labels", 100
    image_parts, label_parts = [], []
    for filename in filenames:
        with (directory / filename).open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        data_value = payload.get("data", payload.get(b"data"))
        label_value = payload.get(label_name, payload.get(label_name.encode("ascii")))
        images = np.asarray(data_value, dtype=np.uint8)
        labels = np.asarray(label_value, dtype=np.int64)
        if images.ndim != 2 or images.shape[1] != 3072 or labels.shape != (images.shape[0],):
            raise ValueError(f"invalid CIFAR batch: {filename}")
        image_parts.append(torch.from_numpy(images.copy()).reshape(-1, 3, 32, 32))
        label_parts.append(torch.from_numpy(labels.copy()).to(torch.long))
    images, labels = torch.cat(image_parts), torch.cat(label_parts)
    if bool(torch.any((labels < 0) | (labels >= expected_classes))):
        raise ValueError("CIFAR labels lie outside the expected range")
    return images.contiguous(), labels.contiguous()
