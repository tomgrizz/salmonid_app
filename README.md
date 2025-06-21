# Salmonid Gradio App

This is a self-contained Gradio application for salmonid tracking.

## Running the application

### With Docker

Build the Docker image:
```bash
docker build -t salmonid-gradio .
```

Run the Docker container:
```bash
docker run -p 7860:7860 salmonid-gradio
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