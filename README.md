# Multiclass output-space PAC-Bayes experiment

This folder contains the experiment code and preset JSON configurations for MNIST, CIFAR-10, CIFAR-100, ImageNet-1k, and ImageNet-to-CIFAR transfer. `run.py` performs the complete procedure: split the training set into A and B, construct the prior and feature map using A, select the posterior using B, certify it using fresh Monte Carlo draws, then evaluate the test set for diagnostics. The stochastic model and certificate calculations are shared across datasets.

## Install and check

```bash
python -m pip install -e .
python run.py --smoke --device cpu --output smoke.json
```

## Run an experiment

Select a file from `configs/` with `--preset NAME` or pass a modified JSON file with `--config PATH`. Paths, device, worker count, and seed are runtime arguments. Examples:

```bash
python run.py --preset mnist-cnn32 --data-root /data/mnist --download \
  --seed 7 --device cuda --output result.json

python run.py --preset cifar10-wrn28-4 --data-root /data/cifar10 \
  --seed 7 --device cuda --workers 4 --output result.json

torchrun --standalone --nproc_per_node=2 run.py --preset imagenet-resnet18 \
  --data-root /data/imagenet --seed 7 --device cuda --workers 8 \
  --save-prior-checkpoint /outputs/imagenet-prior.pt \
  --output /outputs/imagenet-result.json
```

The optional `--save-prior-checkpoint` writes the trained A-only prior immediately after training. A later run can use `--prior-checkpoint PATH` with the same seed and A/B split; the loader compares the saved A indices directly with the current split. This avoids repeating the long ImageNet training if a downstream step is interrupted.

For `cifar10-imagenet-r18` and `cifar100-imagenet-r18`, supply the upstream ImageNet PCA artifact with `--upstream-stats PATH`. If the corresponding ResNet-18 weights are not in the torchvision cache, supply them with `--encoder-weights PATH`. The user must pair the PCA artifact with the weights used to compute it.

Each run writes one JSON file with `config`, `metrics`, and `report`. The reported raw Gaussian KL and output-space quotient KL are computed from the same selected posterior. The direct stochastic-prior holdout bound uses a separate Monte Carlo endpoint followed by the fixed-function binary-KL concentration step.
