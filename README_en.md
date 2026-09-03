# Watermark Remove

[简体中文](README.md) | [English](README_en.md)

<div align="center">
  <img src="design/icon_1024.PNG" alt="Watermark Remove" width="128" height="128">
</div>

> A desktop tool for hard-subtitle and text-watermark removal.

![License](https://img.shields.io/badge/License-Apache--2.0-red.svg)
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-green.svg)

## Overview

Watermark Remove is an AI-assisted desktop application for processing hard subtitles, text watermarks, and solid subtitle panels in media that you are authorized to edit. It provides Chinese and English interfaces and improves local inpainting, mask compositing, small-text detection, and desktop workflow over the upstream project.

## Features

- Text-watermark and hard-subtitle processing for videos and images
- One or more user-drawn processing regions
- Precise OCR detection with small-subtitle enhancement
- Dedicated restoration path for solid or nearly solid subtitle panels
- Basic and Enhanced processing profiles
- CUDA, DirectML, and CPU environment detection with a hardware-acceleration switch
- Multiple tasks, timeline ranges, live progress, and processing logs
- Original pixels are preserved outside the active mask to reduce black boxes and color artifacts

## Quick Start

Python 3.12 and an isolated virtual environment are recommended:

```powershell
python -m venv videoEnv
videoEnv\Scripts\activate
pip install -r requirements.txt
python gui.py
```

NVIDIA users need a CUDA-enabled PyTorch build compatible with their drivers. CPU and DirectML environments need their corresponding runtime packages. OCR and inpainting models may need to be downloaded or made available locally on first use.

## Usage

1. Add a video or image.
2. Draw one or more regions around subtitles or text watermarks.
3. Use **Basic** for standard scenes or **Enhanced** for moving and complex backgrounds.
4. Verify hardware acceleration, then start processing.

Only process media that you own or are authorized to edit.

## Upstream Reference and Credits

This project is a derivative work of [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover). It retains the upstream open-source license and applicable notices while adding a redesigned desktop UI, simplified configuration, and subtitle/watermark pipeline improvements.

The upstream project uses the Apache License 2.0. This repository is also distributed under the included [Apache License 2.0](LICENSE) and is not affiliated with the upstream author or distribution channels.

## License

This repository is released under the [Apache License 2.0](LICENSE).
