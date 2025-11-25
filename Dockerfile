FROM python:3.9-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy configuration files
COPY pyproject.toml README.md ./

# Copy source code
COPY src ./src

# Install the package
RUN pip install --no-cache-dir .

# Set the entrypoint
ENTRYPOINT ["bsort"]
