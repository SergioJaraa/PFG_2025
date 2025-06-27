# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install system dependencies required for PyTorch extensions
RUN apt-get update && apt-get install -y \
    ninja-build \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensuring compatibility with GPU architecture (e.g., Turing: 7.5 for RTX 2060)
ENV TORCH_CUDA_ARCH_LIST="7.5"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
