# 442OOLS: OpenCV 5 & AWS Graviton / x86 Multi-Arch Container
FROM python:3.10-slim-bullseye

# Install system dependencies for OpenCV 5, video codecs, and OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Create video cache directories
RUN mkdir -p input_videos output_videos pitch_calibration frontend

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV AWS_DEFAULT_REGION=us-east-1

CMD ["uvicorn", "server.py:app", "--host", "0.0.0.0", "--port", "8000"]
