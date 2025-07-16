FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY . /app

# Copy botsort weights
COPY botsort_weights /app/botsort_weights

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["python", "gradio_app.py"] 
