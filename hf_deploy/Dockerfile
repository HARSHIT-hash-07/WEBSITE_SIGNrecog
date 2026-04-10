FROM python:3.10-slim

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create a non-root user (Hugging Face Space requirement)
# HF Spaces run with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to /app
WORKDIR /app

# Create output directory for videos
# Since we are running as 'user', we need to ensure permissions
# We'll use /tmp if /app/backend/output is not writable, 
# but let's try to make it work here.
USER root
RUN mkdir -p /app/backend/output /app/weights_hq && chmod -R 777 /app/backend/output /app/weights_hq
USER user

# Set the port (Hugging Face default is 7860)
ENV PORT=7860
EXPOSE 7860

# Run the application
# We use uvicorn to serve the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
