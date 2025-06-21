FROM python:3.10
WORKDIR /app

# Install system dependencies for OpenCV and ML
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Copy botsort weights
COPY botsort_weights /app/botsort_weights

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["python", "gradio_app.py"] 
