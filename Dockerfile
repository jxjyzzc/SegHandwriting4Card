# 1. 使用官方预装CUDA和cuDNN的镜像
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:latest-dev

# 2. 设置工作目录并配置镜像源
WORKDIR /app
RUN sed -i 's|http://.*archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://.*security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 3. 安装必要的编译和运行依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    gcc \
    g++ \
    make \
    wget \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. 安装CUDA工具链（仅安装编译所需的组件）
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y \
        cuda-compiler-12-0 \
        cuda-libraries-dev-12-0 \
        cuda-driver-dev-12-0 \
        cuda-cudart-dev-12-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# 5. 设置CUDA环境变量
ENV PATH=/usr/local/cuda-12.0/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-12.0
ENV CUDAToolkit_ROOT=/usr/local/cuda-12.0

# 6. 克隆PaddlePaddle源码
RUN git config --global http.sslVerify false && \
    git clone https://gitee.com/paddlepaddle/Paddle.git && \
    cd Paddle && \
    git checkout develop

# 7. 编译和安装PaddlePaddle
RUN cd Paddle && \
    mkdir -p build && \
    cd build && \
    cmake .. -GNinja \
        -DPY_VERSION=3.10 \
        -DWITH_GPU=ON \
        -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DON_INFER=ON \
        -DWITH_PYTHON=ON \
        -DWITH_MKLDNN=ON \
        -DWITH_DISTRIBUTE=OFF \
        -DWITH_PSCORE=OFF && \
    ninja -j$(($(nproc)/2)) && \
    cd python/dist && \
    pip install paddlepaddle_gpu-*.whl

# 8. 安装项目依赖
COPY . /app/
RUN pip install --no-cache-dir -r work/requirement.txt && \
    chmod +x /app/entrypoint.sh

# 9. 启动命令
CMD ["/bin/bash"]