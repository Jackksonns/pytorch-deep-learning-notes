# 二维 CNN 基础：从卷积窗口到图像分类

## 1. 模型简介

本模块单独演示二维卷积的基本计算，以及 `Conv2d → ReLU → 池化 → 分类` 的最小流程。

## 2. 核心原理

`corr2d()` 让卷积核在图像的高、宽两个方向滑动，计算局部区域与核的逐元素乘积之和。`TinyCNN` 使用 PyTorch 的 `Conv2d` 学习多个核；池化压缩空间尺寸，分类头输出每张图的两个类别分数。

本模块输入是 `(batch, channel, height, width)`。[`TextCNN`](../../NLP_Basics/NLP_Basics.md) 输入词元序列，经词嵌入后沿时间轴做一维卷积；两者的输入形状与卷积方向不同。

## 3. 环境与运行

从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python 经典网络架构/CNN_Basics/CNN.py --demo
```
