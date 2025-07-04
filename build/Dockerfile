# Use official PaddlePaddle GPU image with CUDA 11.8
FROM paddlepaddle/paddle:2.4.2-gpu-cuda11.6-cudnn8.4

# Set environment variables
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Configure apt mirror and install OpenCV dependencies
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Configure pip mirror and install requirements
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -r work/requirement.txt

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Default command
CMD ["/bin/bash"]
