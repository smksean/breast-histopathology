# Use Python 3.10 slim image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU build) pinned with matching versions
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir \
    Pillow \
    tqdm \
    "fastapi==0.112.2" \
    "uvicorn[standard]==0.30.6" \
    "python-multipart==0.0.9" \
    requests \
    "numpy<2.0"

# Copy your code
COPY . .

# Expose Fly.io’s required port
EXPOSE 8080

# Start the FastAPI server on port 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]



