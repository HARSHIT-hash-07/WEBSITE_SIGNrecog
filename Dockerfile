FROM python:3.10-slim

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements from the hf_deploy folder and install
COPY hf_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from hf_deploy into /app
COPY hf_deploy/ .

# Create a non-root user (Hugging Face Space requirement)
RUN useradd -m -u 1000 user
USER root
RUN mkdir -p /app/backend/output /app/weights_hq /app/weights && \
    chmod -R 777 /app/backend/output /app/weights_hq /app/weights
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR /app
EXPOSE 7860

# Run the application using the backend package inside /app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
