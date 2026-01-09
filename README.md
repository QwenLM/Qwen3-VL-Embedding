<p align="center">
    <img src="https://model-demo.oss-cn-hangzhou.aliyuncs.com/Qwen3-VL-Embedding.png" width="400"/>
    <img src="https://model-demo.oss-cn-hangzhou.aliyuncs.com/Qwen3-VL-Reranker.png" width="400"/>
</p>

# Qwen3-VL-Embedding & Qwen3-VL-Reranker

<!-- Badges section -->
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/QwenLM/Qwen3-VL-Embedding)
[![Hugging Face - Embedding](https://img.shields.io/badge/🤗-Embedding-yellow)](https://huggingface.co/collections/Qwen/qwen3-vl-embedding)
[![Hugging Face - Reranker](https://img.shields.io/badge/🤗-Reranker-yellow)](https://huggingface.co/collections/Qwen/qwen3-vl-reranker)
[![ModelScope - Embedding](https://img.shields.io/badge/ModelScope-Embedding-blue)](https://modelscope.cn/organization/qwen/qwen3-vl-embedding)
[![ModelScope - Reranker](https://img.shields.io/badge/ModelScope-Reranker-blue)](https://modelscope.cn/organization/qwen/qwen3-vl-reranker)
[![Technical Report](https://img.shields.io/badge/📄-Technical%20Report-red)](assets/qwen3vlembedding_technical_report.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
<!-- Brief description -->
**State-of-the-art multimodal embedding and reranking models built on Qwen3-VL, supporting text, images, screenshots, videos, and mixed-modal inputs for advanced information retrieval and cross-modal understanding.**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Model Performance](#model-performance)
- [Citation](#citation)

---

## Overview

The Qwen3-VL-Embedding and Qwen3-VL-Reranker model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl) foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities.

Building on the success of our text-oriented [Qwen3-Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding) and [Qwen3-Reranker](https://huggingface.co/collections/Qwen/qwen3-reranker) series, these multimodal models extend best-in-class performance to visual and video understanding tasks. The models work in tandem: the Embedding model handles the initial recall stage by generating semantically rich vectors, while the Reranking model manages the re-ranking stage with precise relevance scoring, significantly enhancing final retrieval accuracy.

---

## Features

- **🎨 Multimodal Versatility**: Seamlessly process inputs containing text, images, screenshots, and video within a unified framework. Achieve state-of-the-art performance across diverse tasks including image-text retrieval, video-text matching, visual question answering (VQA), and multimodal content clustering.

- **🔄 Unified Representation Space**: Leverage the Qwen3-VL architecture to generate semantically rich vectors that capture both visual and textual information in a shared space, facilitating efficient similarity estimation and retrieval across different modalities.

- **🎯 High-Precision Reranking**: The reranking model accepts input pairs (Query, Document)—where both can consist of arbitrary single or mixed modalities—and outputs precise relevance scores for superior retrieval accuracy.

- **🌍 Exceptional Practicality**: 
  - Support for over 30 languages, ideal for global applications
  - Customizable instructions for task-specific optimization
  - Flexible vector dimensions with Matryoshka Representation Learning (MRL)
  - Strong performance with quantized embeddings for efficient deployment
  - Easy integration into existing retrieval pipelines

---

## Model Architecture

### Model Specifications

| Model | Size | Layers | Sequence Length | Embedding Dimension | Quantization Support | MRL Support | Instruction Aware |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL-Embedding-2B** | 2B | 28 | 32K | 2048 | ✅ | ✅ | ✅ |
| **Qwen3-VL-Embedding-8B** | 8B | 36 | 32K | 4096 | ✅ | ✅ | ✅ |
| **Qwen3-VL-Reranker-2B** | 2B | 28 | 32K | - | - | - | ✅ |
| **Qwen3-VL-Reranker-8B** | 8B | 36 | 32K | - | - | - | ✅ |

### Architecture Design

**Qwen3-VL-Embedding: Dual-Tower Architecture**
- Receives single-modal or mixed-modal input and maps it into a high-dimensional semantic vector
- Extracts the hidden state vector corresponding to the `[EOS]` token from the base model's last layer as the final semantic representation
- Enables efficient, independent encoding necessary for large-scale retrieval

**Qwen3-VL-Reranker: Single-Tower Architecture**
- Receives an input pair `(Query, Document)` and performs pointwise reranking
- Utilizes Cross-Attention mechanism for deeper, finer-grained inter-modal interaction and information fusion
- Expresses relevance score by predicting the generation probability of special tokens (`yes` and `no`)

### Feature Comparison

| | Qwen3-VL-Embedding | Qwen3-VL-Reranker |
|---------|-------------------|-------------------|
| **Core Function** | Semantic Representation, Embedding Generation | Relevance Scoring, Pointwise Re-ranking |
| **Input** | Single modality or mixed modalities | (Query, Document) pair with single- or mixed-modal inputs |
| **Architecture** | Dual-Tower | Single-Tower |
| **Mechanism** | Efficient Retrieval | Deep Inter-Modal Interaction, Precise Alignment |
| **Output** | Semantic Vector | Relevance Score |

Both models are built through a multi-stage training paradigm that fully leverages the powerful general multimodal semantic understanding capabilities of Qwen3-VL, providing high-quality semantic representations and precise re-ranking mechanisms for complex, large-scale multimodal retrieval tasks.

---

## Installation

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
cd Qwen3-VL-Embedding

# Run the script to setup the environment
bash scripts/setup_environment.sh
```

The setup script will automatically:
- Install `uv` if not already installed
- Install all project dependencies

After setup completes, activate the environment:
```bash
source .venv/bin/activate
```

### Download Models

Our models are available on both Hugging Face and ModelScope.

| Model | Hugging Face | ModelScope |
|-------|--------------|------------|
| Qwen3-VL-Embedding-2B |[Link](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [Link](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B) |
| Qwen3-VL-Embedding-8B |[Link](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | [Link](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-8B) |
| Qwen3-VL-Reranker-2B |[Link](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | [Link](https://modelscope.cn/models/qwen/Qwen3-VL-Reranker-2B) |
| Qwen3-VL-Reranker-8B |[Link](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | [Link](https://modelscope.cn/models/qwen/Qwen3-VL-Reranker-8B) |

**Install download dependencies:**

**Download from Hugging Face:**
```bash
uv pip install huggingface-hub

huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B
```

**Download from ModelScope:**
```bash
uv pip install modelscope

modelscope download --model qwen/Qwen3-VL-Embedding-2B --local_dir ./models/Qwen3-VL-Embedding-2B
```

## Usage

### Quick Start

#### Embedding Model

##### Transformers usage

```python
import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

model = Qwen3VLEmbedder(
    model_name_or_path="./models/Qwen3-VL-Embedding-2B",
    # flash_attention_2 for better acceleration and memory saving
    # torch_dtype=torch.bfloat16, 
    # attn_implementation="flash_attention_2"
)

inputs = [{
    "text": "A woman playing with her dog on a beach at sunset.",
    "instruction": "Retrieve images or text relevant to the user's query.",
}, {
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
}, {
    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
}, {
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
}]

embeddings = model.process(inputs)
print(embeddings @ embeddings.T)
```

##### vLLM usage

```python
# Requires vllm>=0.14.0
from io import BytesIO

import requests
import torch
from PIL import Image

from vllm import LLM


def get_image_from_url(url) -> Image.Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


model = LLM(model="Qwen/Qwen3-VL-Embedding-2B", runner="pooling")

image = get_image_from_url("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
inputs = [
    {
        "prompt": "A woman playing with her dog on a beach at sunset.",
    },
    {
        "prompt": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
    },
    {
        "prompt": image_placeholder,
        "multi_modal_data": {"image": image},
    },
    {
        "prompt": f"{image_placeholder}\nA woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
        "multi_modal_data": {"image": image},
    },
]

outputs = model.embed(inputs)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = embeddings[:2] @ embeddings[2:].T
print(scores.tolist())
```

#### Reranking Model

##### Transformers usage

```python
import torch
from src.models.qwen3_vl_reranker import Qwen3VLReranker

model = Qwen3VLReranker(
    model_name_or_path="./models/Qwen3-VL-Reranker-2B",
    # flash_attention_2 for better acceleration and memory saving
    # torch_dtype=torch.bfloat16, 
    # attn_implementation="flash_attention_2"
)

inputs = {
    "instruction": "Retrieve images or text relevant to the user's query.",
    "query": {"text": "A woman playing with her dog on a beach at sunset."},
    "documents": [
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    ],
    "fps": 1.0, 
    "max_frames": 64
}

scores = model.process(inputs)
print(scores)
```

##### vLLM usage

```python
# Requires vllm>=0.14.0
from vllm import LLM

model = LLM(
    model="Qwen/Qwen3-VL-Reranker-2B",
    runner="pooling",
    hf_overrides={
        "architectures": ["Qwen3VLForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
)

query = "A woman playing with her dog on a beach at sunset."
# Sample multimodal documents to be scored against the query
# Each document contains an image URL that will be fetched and processed
documents = {
    "content": [
        {
            "type": "text",
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
        },
        {
            "type": "video_url",
            "video_url": {
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
            },
        },
    ]
}
outputs = model.score(query, documents)
print("Relevance scores:", [output.outputs.score for output in outputs])
```

### Model Input Specification

#### Multimodal Object
A dictionary that can contain the following keys:
- **text**: Text input as a string
- **image**: Image input, supports:
  - Local file path
  - URL (network path)
  - `PIL.Image.Image` instance
- **video**: Video input, supports:
  - Local file path
  - URL (network path)
  - Sequence of video frames

#### Instruction
Task description for relevance evaluation (default: "Represent the user's input")

#### Video Sampling Settings
Only effective when video input is a video file:
- **fps**: Frame sampling rate per second (frames per second)
- **max_frames**: Maximum number of frames to sample

#### Input Format

**Embedding Model**: A list of dictionaries, where each dictionary contains:
- Instruction (optional)
- Video sampling settings (optional)
- Multimodal object keys (text, image, and/or video)

**Reranking Model**: A dictionary containing:
- **query**: A multimodal object
- **documents**: A list of multimodal objects
- **instruction**: Task description (optional)
- **fps**: Video sampling rate (optional)
- **max_frames**: Maximum frames (optional)

### Embedding Model

#### Model Initialization Parameters

```python
Qwen3VLEmbedder(
    model_name_or_path="./models/Qwen3-VL-Embedding-2B",
    max_length=8192,           # Default context length
    min_pixels=4096,           # Minimum pixels for input images
    max_pixels=1843200,        # Maximum pixels for input images (equivalent to 1280×1440 resolution)
    total_pixels=7864320,      # Maximum total pixels for input videos (multiplied by 2 in model)
                              # For a 16-frame video, each frame can have up to 983040 pixels (1280×768 resolution)
    fps=1.0,                   # Default sampling frame rate for video files (frames per second)
    num_frames=64,             # Default number of frames when video input is a frame sequence
    max_frames=64,             # Maximum number of frames for video input
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

---

## Examples

### Embedding Model

We provide comprehensive examples in `examples/embedding.ipynb` demonstrating various tasks across different modalities:

**Text Tasks:**
- Text Classification (AG News)
- Text Question Answering (SQuAD)
- Text Retrieval (MS MARCO)

**Image Tasks:**
- Image Classification (CIFAR-10)
- Image Question Answering (VQAv2)
- Image Retrieval (MS COCO)

Examples for video and visual document tasks are presented in the appendix of [technical report](assets/qwen3vlembedding_technical_report.pdf)

We also provide an end-to-end multimodal RAG example using Qwen3-VL-Embedding, Qwen3-VL-Reranker and Qwen3-VL [here](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/examples/Qwen3VL_Multimodal_RAG.ipynb).

### Reranking Model

Coming soon.

---

## Model Performance

### Embedding Model

#### Evaluation Results on [MMEB-V2](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)

Results on the MMEB-V2 benchmark. All models except IFM-TTE have been re-evaluated on the updated VisDoc OOD split. CLS: classification, QA: question answering, RET: retrieval, GD: grounding, MRET: moment retrieval, VDR: ViDoRe, VR: VisRAG, OOD: out-of-distribution.

| Model                      | Model Size | Image CLS | Image QA | Image RET | Image GD | Image Overall | Video CLS | Video QA | Video RET | Video MRET | Video Overall | VisDoc VDRv1 | VisDoc VDRv2 | VisDoc VR | VisDoc OOD | VisDoc Overall | All    |
|----------------------------|---------|-------|------|------|------|-----------|------|------|------|------|------|-------|------|--------|------|------|--------|
| **# of Datasets →**        |         | 10    | 10   | 12   | 4    | 36        | 5    | 5    | 5    | 3    | 18   | 10    | 4    | 6      | 4    | 24   | 78     |
| VLM2Vec                    | 2B      | 58.7 | 49.3 | 65.0 | 72.9 | 59.7 | 33.4 | 30.5 | 20.6 | 30.7 | 28.6 | 49.8 | 13.5 | 51.8 | 48.2 | 44.0 | 47.7 |
| VLM2Vec-V2                 | 2B      | 62.9 | 56.3 | 69.5 | 77.3 | 64.9 | 39.3 | 34.3 | 28.8 | 36.8 | 34.6 | 75.5 | 44.9 | 79.4 | 62.2 | 69.2 | 59.2 |
| GME-2B                     | 2B      | 54.4 | 29.9 | 66.9 | 55.5 | 51.9 | 34.9 | 42.0 | 25.6 | 31.1 | 33.6 | 86.1 | 54.0 | 82.5 | 67.5 | 76.8 | 55.3 |
| GME-7B                     | 7B      | 57.7 | 34.7 | 71.2 | 59.3 | 56.0 | 37.4 | 50.4 | 28.4 | 37.0 | 38.4 | 89.4 | 55.6 | 85.0 | 68.3 | 79.3 | 59.1 |
| Ops-MM-embedding-v1        | 8B      | 69.7 | 69.6 | 73.1 | 87.2 | 72.7 | 59.7 | 62.2 | 45.7 | 43.2 | 53.8 | 80.1 | 59.6 | 79.3 | 67.8 | 74.4 | 68.9 |
| IFM-TTE                    | 8B      | 76.7 | 78.5 | 74.6 | 89.3 | 77.9 | 60.5 | 67.9 | 51.7 | 54.9 | 59.2 | 85.2 | 71.5 | 92.7 | 53.3 | 79.5 | 74.1 |
| RzenEmbed                  | 8B      | 70.6 | 71.7 | 78.5 | 92.1 | 75.9 | 58.8 | 63.5 | 51.0 | 45.5 | 55.7 | 89.7 | 60.7 | 88.7 | 69.9 | 81.3 | 72.9 |
| Seed-1.6-embedding-1215    | unknown | 75.0 | 74.9 | 79.3 | 89.0 | 78.0 | 85.2 | 66.7 | 59.1 | 54.8 | 67.7 | 90.0 | 60.3 | 90.0 | 70.7 | 82.2 | 76.9 | 
| **Qwen3-VL-Embedding-2B**  | 2B      | 70.3 | 74.3 | 74.8 | 88.5 | 75.0 | 71.9 | 64.9 | 53.9 | 53.3 | 61.9 | 84.4 | 65.3 | 86.4 | 69.4 | 79.2 | 73.2 |
| **Qwen3-VL-Embedding-8B**  | 8B      | 74.2 | 81.1 | 80.0 | 92.2 | 80.1 | 78.4 | 71.0 | 58.7 | 56.1 | 67.1 | 87.2 | 69.9 | 88.7 | 73.3 | 82.4 | **77.8** |

#### Evaluation Results on [MMTEB](https://huggingface.co/spaces/mteb/leaderboard)

Results on the MMTEB benchmark. 

| Model                            |  Size   |  Mean (Task)  | Mean (Type) | Bitxt Mining | Class. | Clust. | Inst. Retri. | Multi. Class. | Pair. Class. | Rerank | Retri. | STS  |
|----------------------------------|:-------:|:-------------:|:-------------:|:--------------:|:--------:|:--------:|:--------------:|:---------------:|:--------------:|:--------:|:--------:|:------:|
| NV-Embed-v2                      |   7B    |     56.29     | 49.58       | 57.84        | 57.29  | 40.80  | 1.04         | 18.63         | 78.94        | 63.82  | 56.72  | 71.10|
| GritLM-7B                        |   7B    |     60.92     | 53.74       | 70.53        | 61.83  | 49.75  | 3.45         | 22.77         | 79.94        | 63.78  | 58.31  | 73.33|
| BGE-M3                           |  0.6B   |     59.56     | 52.18       | 79.11        | 60.35  | 40.88  | -3.11        | 20.1          | 80.76        | 62.79  | 54.60  | 74.12|
| multilingual-e5-large-instruct   |  0.6B   |     63.22     | 55.08       | 80.13        | 64.94  | 50.75  | -0.40        | 22.91         | 80.86        | 62.61  | 57.12  | 76.81|
| gte-Qwen2-1.5B-instruct          |  1.5B   |     59.45     | 52.69       | 62.51        | 58.32  | 52.05  | 0.74         | 24.02         | 81.58        | 62.58  | 60.78  | 71.61|
| gte-Qwen2-7b-Instruct            |   7B    |     62.51     | 55.93       | 73.92        | 61.55  | 52.77  | 4.94         | 25.48         | 85.13        | 65.55  | 60.08  | 73.98|
| text-embedding-3-large           |    -    |     58.93     | 51.41       | 62.17        | 60.27  | 46.89  | -2.68        | 22.03         | 79.17        | 63.89  | 59.27  | 71.68|
| Cohere-embed-multilingual-v3.0   |    -    |     61.12     | 53.23       | 70.50        | 62.95  | 46.89  | -1.89        | 22.74         | 79.88        | 64.07  | 59.16  | 74.80|
| Gemini Embedding                 |    -    |     68.37     | 59.59       | 79.28        | 71.82  | 54.59  | 5.18         | **29.16**     | 83.63        | 65.58  | 67.71  | 79.40|
| Qwen3-Embedding-0.6B        |  0.6B   |     64.33     | 56.00       | 72.22        | 66.83  | 52.33  | 5.09         | 24.59         | 80.83        | 61.41  | 64.64  | 76.17|
| Qwen3-Embedding-4B           |   4B    |     69.45     | 60.86       | 79.36        | 72.33  | 57.15  | **11.56**    | 26.77         | 85.05        | 65.08  | 69.60  | 80.86|
| Qwen3-Embedding-8B          |   8B    |   **70.58**   | **61.69**   | **80.89**    | **74.00** | **57.65** | 10.06      | 28.66         | **86.40**    | **65.63** | **70.88** | **81.08** |
| Qwen3-VL-Embedding-2B | 2B | 63.87 | 55.84 | 69.51 | 65.86 | 52.50 | 3.87 | 26.08 | 78.50 | 64.80 | 67.12 | 74.29 |
| Qwen3-VL-Embedding-8B | 8B | 67.88 | 58.88 | 77.48 | 71.95 | 55.82 | 4.46 | 28.59 | 81.08 | 65.72 | 69.41 | 75.41 |

### Reranking Model

We utilize retrieval task datasets from various subtasks of [MMEB-v2](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard) and [MMTEB](https://huggingface.co/spaces/mteb/leaderboard) retrieval benchmarks. For visual document retrieval, we employ [JinaVDR](https://huggingface.co/collections/jinaai/jinavdr-visual-document-retrieval) and [ViDoRe v3](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3) datasets. Our results demonstrate that all Qwen3-VL-Reranker models consistently outperform the base embedding model and baseline rerankers, with the 8B variant achieving the best performance across most tasks.

| Model | Size | MMEB-v2(Retrieval) - Avg | MMEB-v2(Retrieval) - Image | MMEB-v2(Retrieval) - Video | MMEB-v2(Retrieval) - VisDoc | MMTEB(Retrieval) | JinaVDR | ViDoRe(v3) |
|-------|------|--------------------------|----------------------------|----------------------------|------------------------------|------------------|---------|------------|
| Qwen3-VL-Embedding-2B | 2B | 73.4 | 74.8 | 53.6 | 79.2 | 68.1 | 71.0 | 52.9 |
| jina-reranker-m0      | 2B |  - | 68.2 | -    | 85.2 | -    | 82.2 | 57.8 |
| Qwen3-VL-Reranker-2B | 2B | 75.1 | 73.8 | 52.1 | 83.4 | 70.0 | 80.9 | 60.8 |
| Qwen3-VL-Reranker-8B | 8B | 79.2 | 80.7 | 55.8 | 86.3 | 74.9 | 83.6 | 66.7 |

### Reproducing Evaluation

#### Embedding Model

We provide reproducible evaluation code for **MMEB v2** benchmark, based on [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). To reproduce the results:

1. **Download evaluation data:**
   ```bash
   bash data/evaluation/mmeb_v2/download_data.sh
   ```

2. **Run evaluation:**
   ```bash
   bash scripts/evaluation/mmeb_v2/eval_embedding.sh
   ```
   Run the script without arguments to see the required parameters. The script will evaluate tasks and collect results automatically.

#### Reranking Model

Coming soon.

---

## Citation

```bibtex
@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen, Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv},
  year={2026}
}
```
