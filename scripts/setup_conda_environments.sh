#!/bin/bash

# 检测并安装对应版本的PyTorch
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "检测到 NVIDIA GPU，CUDA 版本: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" > "12.9" ]] || [[ "$CUDA_VERSION" == "12.9"* ]]; then
        echo "安装 CUDA 12.9 版本..."
        pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
    elif [[ "$CUDA_VERSION" == "12.8"* ]]; then
        echo "安装 CUDA 12.8 版本..."
        pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    else
        echo "安装 CUDA 12.6 版本（兼容 12.6+）..."
        pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    fi
    pip install faiss-gpu-cu12
else
    echo "未检测到 GPU，安装 CPU 版本..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    pip install faiss-cpu
fi