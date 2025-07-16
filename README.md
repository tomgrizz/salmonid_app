# Salmonid Gradio App

This is a self-contained Gradio application for salmonid tracking with GPU acceleration support.

## GPU Setup (Recommended)

For optimal performance, the app supports GPU acceleration using CUDA. To enable GPU support:

### 1. Check GPU Compatibility
Run the GPU test script to verify your setup:
```bash
python test_gpu.py
```

### 2. Install CUDA-enabled PyTorch
If you don't have CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify GPU Detection
The app will automatically detect and use available GPUs. You should see GPU information when starting the app.

## Running the application

### With Docker (GPU Support)

#### Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```

#### Option 2: Manual Docker Commands
Build the Docker image with GPU support:
```bash
docker build -t salmonid-gradio .
```

Run the Docker container with GPU access:
```bash
docker run --gpus all -p 7860:7860 salmonid-gradio
```

### Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
python gradio_app.py
```

## Performance Notes

- **GPU Mode**: Significantly faster processing, especially for longer videos
- **CPU Mode**: Works on all systems but slower processing
- The app automatically falls back to CPU if GPU is not available
- GPU memory usage scales with video resolution and batch size 