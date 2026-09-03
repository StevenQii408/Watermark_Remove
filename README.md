# Watermark Remove

[简体中文](README.md) | [English](README_en.md)


> 面向桌面端的视频硬字幕与文字水印处理工具。

![License](https://img.shields.io/badge/License-Apache--2.0-red.svg)
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-green.svg)

## 项目简介

Watermark Remove 是一个基于 AI 视频修复能力的桌面工具，用于处理用户有权编辑的视频中的硬字幕、文字水印及其纯色底板。项目提供中文和英文界面，并在原有能力上针对小字幕、纯色字幕底、局部修复、掩码融合和桌面交互进行了优化。

## 主要功能

- 视频和图片的文字水印/硬字幕处理
- 鼠标框选一个或多个处理区域
- 精细 OCR 检测与小字幕增强检测
- 纯色或近似纯色字幕底板的专用恢复路径
- 基础版与增强版两种处理配置
- CUDA、DirectML 与 CPU 环境检测及硬件加速开关
- 多任务队列、时间轴选段、实时进度与处理日志
- 掩码外区域保持原始画面，降低黑框和颜色混合问题

## 快速开始

建议使用 Python 3.12 及独立虚拟环境：

```powershell
python -m venv videoEnv
videoEnv\Scripts\activate
pip install -r requirements.txt
python gui.py
```

NVIDIA GPU 用户需要安装与驱动匹配的 CUDA 版 PyTorch；CPU 或 DirectML 环境可按项目依赖说明安装对应运行时。首次启动时，OCR 与修复模型可能需要下载或准备本地模型文件。

## 使用建议

1. 添加视频或图片。
2. 在预览中框选字幕或文字水印区域。
3. 基础场景选择“基础版”；运动背景或复杂画面选择“增强版”。
4. 确认硬件加速状态后开始处理。

请仅处理你拥有编辑权或已获得授权的媒体内容。

## 参考与致谢

本项目基于 [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) 进行二次开发。保留了其原有的开源许可与相关版权声明，并在此基础上进行了界面重构、配置简化及字幕/水印处理链路优化。

源项目采用 Apache License 2.0；本项目同样遵循仓库中的 [Apache License 2.0](LICENSE)。本项目与源项目作者及其发布渠道没有隶属关系。

## 许可证

本仓库代码以 [Apache License 2.0](LICENSE) 发布。
