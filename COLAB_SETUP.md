# Google Colab Setup Guide for DDN Training

This guide explains how to run DDN training on Google Colab using the minimal training script.

## Quick Start

### Step 1: Setup Environment

```python
# Cell 1: Clone repository and install dependencies
!git clone https://github.com/EmreDinc10/DDN-training-classic.git
%cd DDN-training-classic
!pip install -r requirements.txt
```

### Step 2: Prepare Dataset

**Option A: Use CIFAR-10 (prepared ZIP)**
```python
# Cell 2: Download CIFAR-10 dataset
!mkdir -p datasets
!wget -O datasets/cifar10-32x32.zip https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Note: You may need to convert this to the expected format using dataset_tool.py
```

**Option B: Use your own dataset**
```python
# Cell 2: Upload your dataset ZIP file
from google.colab import files
uploaded = files.upload()  # Upload your dataset.zip file
# Move to datasets folder
!mkdir -p datasets
!mv *.zip datasets/
```

### Step 3: Run Minimal Training

```python
# Cell 3: Run minimal training script
import os

# Set environment variables (optional - defaults are fine)
os.environ["DATA_PATH"] = "datasets/cifar10-32x32.zip"
os.environ["OUTDIR"] = "/content/training-runs/minimal-test"
os.environ["BATCH_SIZE"] = "32"
os.environ["BATCH_GPU"] = "32"
os.environ["TOTAL_KIMG"] = "10"  # 10k images = ~10 minutes

# Run training
!python train_minimal.py
```

## Complete Colab Notebook Template

Copy this into a new Colab notebook:

```python
# ============================================================================
# CELL 1: Setup
# ============================================================================
# Enable GPU: Runtime -> Change runtime type -> GPU

!git clone https://github.com/EmreDinc10/DDN-training-classic.git
%cd DDN-training-classic
!pip install -r requirements.txt

print("✓ Setup complete!")

# ============================================================================
# CELL 2: Prepare Dataset
# ============================================================================
# Option 1: Download CIFAR-10 (if available)
# !mkdir -p datasets
# !wget -O datasets/cifar10-32x32.zip <dataset_url>

# Option 2: Upload your dataset
from google.colab import files
import os
os.makedirs("datasets", exist_ok=True)
# Upload your ZIP file when prompted
uploaded = files.upload()
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        !mv {filename} datasets/

print("✓ Dataset ready!")

# ============================================================================
# CELL 3: Configure Training
# ============================================================================
import os

# Training configuration
config = {
    "DATA_PATH": "datasets/cifar10-32x32.zip",  # Change to your dataset
    "OUTDIR": "/content/training-runs/minimal-test",
    "BATCH_SIZE": "32",      # Total batch size
    "BATCH_GPU": "32",       # Per GPU batch size
    "TOTAL_KIMG": "10",      # Training duration (10 = 10k images, ~10 min)
}

# Set environment variables
for key, value in config.items():
    os.environ[key] = value
    print(f"{key} = {value}")

print("\n✓ Configuration set!")
print(f"Estimated training time: ~{int(config['TOTAL_KIMG']) * 0.5:.1f} minutes")

# ============================================================================
# CELL 4: Run Training
# ============================================================================
!python train_minimal.py

# ============================================================================
# CELL 5: View Results (after training)
# ============================================================================
import os
from IPython.display import Image, display

# List checkpoints
output_dir = os.environ.get("OUTDIR", "/content/training-runs/minimal-test")
print(f"\nCheckpoints saved in: {output_dir}")
!ls -lh {output_dir}/*.pkl 2>/dev/null || echo "No checkpoints found yet"

# If visualization images exist
vis_path = f"{output_dir}/vis.png"
if os.path.exists(vis_path):
    display(Image(vis_path))
```

## Training Parameters Explained

### Quick Test (5-10 minutes)
```python
os.environ["TOTAL_KIMG"] = "5"   # 5k images
os.environ["BATCH_SIZE"] = "32"
```

### Short Training (30-60 minutes)
```python
os.environ["TOTAL_KIMG"] = "50"  # 50k images
os.environ["BATCH_SIZE"] = "64"
```

### Medium Training (2-4 hours)
```python
os.environ["TOTAL_KIMG"] = "200"  # 200k images
os.environ["BATCH_SIZE"] = "128"
```

## Monitoring Progress

The minimal script prints progress every 1k images (`kimg_per_tick=1`). You'll see:
- Training loss
- Progress percentage
- Time elapsed
- Memory usage

## Saving Checkpoints

- Checkpoints saved every 5 ticks (5k images)
- Location: `/content/training-runs/minimal-test/shot-*.pkl`
- To resume: Modify script to use `resume_pkl` parameter

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` and `BATCH_GPU` to 16 or 8
- Reduce model size (not configurable in minimal script)

### Dataset Not Found
- Check dataset path in `DATA_PATH`
- Ensure ZIP file is in `datasets/` folder
- Verify dataset format matches expected structure

### Slow Training
- Colab free tier has limited GPU (T4)
- Consider using Colab Pro for better GPUs
- Reduce `TOTAL_KIMG` for faster testing

## Using Full Training Script

For full training with all options:

```python
!python train.py \
  --data=datasets/cifar10-32x32.zip \
  --outdir=/content/training-runs \
  --batch=32 \
  --batch-gpu=32 \
  --duration=0.1 \
  --tick=5 \
  --snap=10
```

## Next Steps

1. **Generate samples**: Use `generate.py` with trained checkpoint
2. **Evaluate**: Calculate FID score using `fid.py`
3. **Download results**: Use Colab file browser or mount Google Drive

## Mount Google Drive (Optional)

To save results permanently:

```python
from google.colab import drive
drive.mount('/content/drive')

# Use Drive path for output
os.environ["OUTDIR"] = "/content/drive/MyDrive/ddn-training-runs"
```

