# Vision Transformer（ViT）——使用 Transformer 进行图像分类

## 1. 模型简介

ViT 的思路是将图片切成一系列 Patch，把图像当成 Token 序列，使用 Transformer Encoder 完成分类。

本实验从零实现 **Mini-ViT**，用 CIFAR-10 训练，并可在**同一划分**下运行 ResNet18 基线。

## 2. 核心原理

```text
输入 (B, 3, 32, 32)
    ↓ Conv2d(kernel_size=patch_size, stride=patch_size)
Patch Embedding (B, N, D)
    ↓ 拼接 CLS Token + 可学习 Position Embedding
Transformer Encoder × L（无 Causal Mask）
    ↓ 提取 CLS Token
分类 Head → (B, 10)
```

以 `patch_size=4` 为例，`N=(32/4)^2=64`。拼接一个 `[CLS]` Token 后，Encoder 接收 65 个 Token。

注意：ViT 和 GPT 都使用 Multi-Head Attention，但 **ViT 是双向 Encoder Attention**，无需未来信息掩码。

## 3. PyTorch 实现

- `ViT.py`：手写 `PatchEmbedding`、`ViTSelfAttention`、`EncoderBlock` 和 `VisionTransformer`。
- `train.py`：使用标准 CIFAR-10 训练集，固定种子划分训练与验证，并使用独立测试集。
- `train.py --arch resnet18`：在相同数据划分及增强策略下训练 ResNet18。
- `visualize_attention.py`：加载训练得到的 ViT，画出最后一层 CLS→Patch Attention。
- `results/`：保存训练曲线、最终权重和分类预测示例。

其中 `ViT.py` 的 `return_attention=True` 会返回每个 Encoder Block 的注意力权重。

## 4. 环境与运行

```bash
python -m pip install -r requirements.txt
cd 经典网络架构/ViT

# 从零训练 Mini-ViT（首次运行下载 CIFAR-10）
python train.py --arch vit --epochs 20 --patch-size 4 --batch-size 128

# 同一 CIFAR-10 划分的 CNN 基线（独立训练）
python train.py --arch resnet18 --epochs 20 --batch-size 128

# 比较 Patch Size（新的模型与新的训练）
python train.py --arch vit --epochs 20 --patch-size 8 --batch-size 128

# 对 patch=4 的 ViT 绘制注意力示意图
python visualize_attention.py --checkpoint results/vit_p4.pth --index 0
```

PS：默认 `train:valid=90%:10%`。

