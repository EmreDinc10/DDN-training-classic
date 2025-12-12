#!/usr/bin/env python3
"""
Verification script for Colab connection and GPU availability.
Run this in Colab to verify your setup.
"""

import sys
import os

print("="*70)
print("COLAB & GPU VERIFICATION")
print("="*70)

# 1. Verify Colab Connection
print("\n[1] COLAB CONNECTION CHECK")
print("-" * 70)
try:
    import google.colab
    print("✓ Running in Google Colab")
    print(f"✓ Colab module imported successfully")
    colab_connected = True
except ImportError:
    print("✗ NOT running in Google Colab")
    print("  (google.colab module not available)")
    colab_connected = False

# Check current directory
current_dir = os.getcwd()
print(f"\nCurrent directory: {current_dir}")
if "/content" in current_dir:
    print("✓ In Colab filesystem (/content)")
else:
    print("⚠ Not in Colab filesystem")

# 2. Verify GPU/CUDA
print("\n[2] GPU/CUDA CHECK")
print("-" * 70)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = gpu_props.total_memory / 1e9
            
            print(f"\n  GPU {i}:")
            print(f"    Name: {gpu_name}")
            print(f"    Memory: {gpu_memory_gb:.2f} GB")
            print(f"    Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            # Check if it's A100
            if "A100" in gpu_name:
                print(f"    ✓ A100 GPU detected!")
            elif "V100" in gpu_name:
                print(f"    ✓ V100 GPU detected")
            elif "T4" in gpu_name:
                print(f"    ✓ T4 GPU detected")
            
            # Check current device
            if torch.cuda.current_device() == i:
                print(f"    ✓ Currently active device")
        
        # Test CUDA operation
        try:
            test_tensor = torch.randn(10, 10).cuda()
            print(f"\n✓ CUDA operations working (test tensor created on GPU)")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n✗ CUDA operations failed: {e}")
    else:
        print("\n✗ NO GPU DETECTED!")
        print("\n  To enable GPU in Colab:")
        print("  1. Go to: Runtime → Change runtime type")
        print("  2. Set Hardware accelerator to: GPU")
        print("  3. Click Save")
        print("  4. Runtime → Restart runtime")
        print("  5. Run this script again")
        
except ImportError:
    print("✗ PyTorch not installed")
    print("  Install with: !pip install torch")
except Exception as e:
    print(f"✗ Error checking GPU: {e}")

# 3. Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Colab connection: {'✓ Connected' if colab_connected else '✗ Not connected'}")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU status: ✓ Available ({gpu_name})")
        if "A100" in gpu_name:
            print("GPU type: ✓ A100 (Best for training!)")
        else:
            print(f"GPU type: {gpu_name} (Good, but A100 is faster)")
    else:
        print("GPU status: ✗ Not available")
except:
    print("GPU status: ✗ Could not check")

print("="*70)

# Final recommendation
if colab_connected:
    try:
        import torch
        if torch.cuda.is_available():
            print("\n✅ READY FOR TRAINING!")
            print("   Your Colab environment is properly configured.")
        else:
            print("\n⚠ GPU REQUIRED!")
            print("   Enable GPU before running training.")
    except:
        pass
else:
    print("\n⚠ Not running in Colab environment")

