# Minimal DDN Training Script for Quick Testing
# Optimized for Google Colab and quick progress observation

import os
import sys
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

# Initialize distributed (works for single GPU too)
torch.multiprocessing.set_start_method("spawn", force=True)
dist.init()

# Minimal configuration for quick testing
c = dnnlib.EasyDict()

# Dataset configuration
c.dataset_kwargs = dnnlib.EasyDict(
    class_name="training.dataset.ImageFolderDataset",
    path=os.environ.get("DATA_PATH", "datasets/cifar10-32x32.zip"),
    use_labels=False,
    xflip=False,
    cache=True,
)

# Data loader
c.data_loader_kwargs = dnnlib.EasyDict(
    pin_memory=True,
    num_workers=2,
    prefetch_factor=2
)

# Network configuration (DDN)
c.network_kwargs = dnnlib.EasyDict(
    class_name="training.networks.DDNPrecond",
    model_type="PHDDN",
    img_resolution=32,
    in_channels=3,
    out_channels=3,
    label_dim=0,
    augment_dim=0,
    dropout=0.0,
    use_fp16=True,
)

# Loss configuration
c.loss_kwargs = dnnlib.EasyDict(
    class_name="training.loss.DDNLoss"
)

# Optimizer
c.optimizer_kwargs = dnnlib.EasyDict(
    class_name="torch.optim.Adam",
    lr=1e-4,
    betas=[0.9, 0.999],
    eps=1e-8
)

# Training parameters (minimal for quick testing)
c.batch_size = int(os.environ.get("BATCH_SIZE", "32"))
c.batch_gpu = int(os.environ.get("BATCH_GPU", "32"))
c.total_kimg = int(os.environ.get("TOTAL_KIMG", "10"))  # 10k images = ~10 minutes
c.ema_halflife_kimg = 0  # Disable EMA for speed
c.loss_scaling = 1.0
c.cudnn_benchmark = True

# Progress tracking
c.kimg_per_tick = 1  # Print every 1k images
c.snapshot_ticks = 5  # Save checkpoint every 5 ticks
c.state_dump_ticks = 10

# Output directory
c.run_dir = os.environ.get("OUTDIR", "/content/training-runs/minimal-test")
os.makedirs(c.run_dir, exist_ok=True)

# Set DDN-specific parameters
import sddn
sddn.DiscreteDistributionOutput.learn_residual = True
sddn.DiscreteDistributionOutput.chain_dropout = 0.05

# Validate dataset
try:
    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    dataset_name = dataset_obj.name
    c.dataset_kwargs.resolution = dataset_obj.resolution
    c.dataset_kwargs.max_size = len(dataset_obj)
    print(f"Dataset: {dataset_name}, Resolution: {dataset_obj.resolution}, Size: {len(dataset_obj)}")
    del dataset_obj
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Print configuration
print("\n" + "="*60)
print("MINIMAL DDN TRAINING CONFIGURATION")
print("="*60)
print(f"Dataset: {c.dataset_kwargs.path}")
print(f"Output: {c.run_dir}")
print(f"Batch size: {c.batch_size} (per GPU: {c.batch_gpu})")
print(f"Total kimg: {c.total_kimg}")
print(f"Estimated time: ~{c.total_kimg * 0.5:.1f} minutes")
print("="*60 + "\n")

# Run training
training_loop.training_loop(**c)

