# Dynamic LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token

[![arXiv](https://img.shields.io/badge/arXiv-2501.03895-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.03895)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20huggingface%20-llava--mini--llama--3.1--8b-orange.svg)](https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b)

LLaVA-Mini is a unified large multimodal model that can support the understanding of images, high-resolution images, and videos in an efficient manner. Guided by the interpretability within LMM, LLaVA-Mini significantly improves efficiency while ensuring vision capabilities. [Model](https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b) and [demo](#-demo) of LLaVA-Mini are available now!

> [!Note]
> LLaVA-Mini only requires **1 token** to represent each image, which improves the efficiency of image and video understanding, including:
> - **Computational effort**: 77% FLOPs reduction
> - **Response latency**: reduce from 100 milliseconds to 40 milliseconds
> - **VRAM memory usage**: reduce from 360 MB/image to 0.6 MB/image, support 3-hour video processing
