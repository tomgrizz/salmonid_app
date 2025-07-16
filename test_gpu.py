#!/usr/bin/env python3
"""
GPU Test Script for Salmonid Tracking App
This script checks if GPU is available and provides information about the system setup.
"""

import torch
import sys

def test_gpu():
    print("=== GPU Test for Salmonid Tracking App ===\n")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = getattr(torch.version, "cuda", None)
        print(f"CUDA version: {cuda_version}")
        
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get information about each GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU tensor operations
        print("\nTesting GPU tensor operations...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✓ GPU tensor operations successful")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
            return False
        
        # Test model loading on GPU
        print("\nTesting model loading on GPU...")
        try:
            from transformers import AutoModelForObjectDetection
            model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
            model = model.cuda()
            model.eval()
            print("✓ Model loading on GPU successful")
        except Exception as e:
            print(f"✗ Model loading on GPU failed: {e}")
            return False
            
    else:
        print("\nNo CUDA available. The app will run on CPU.")
        print("To enable GPU acceleration:")
        print("1. Install NVIDIA drivers")
        print("2. Install CUDA toolkit")
        print("3. Install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print("\n=== GPU Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1) 