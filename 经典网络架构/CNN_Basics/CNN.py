"""二维 CNN 基础：局部卷积、池化与合成条纹分类。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

# 进一步学习可接入 Fashion-MNIST：
# https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html


def corr2d(image, kernel):
    """单通道二维互相关，帮助理解 Conv2d 的滑动窗口。"""
    height, width = kernel.shape
    output = image.new_zeros(image.size(0) - height + 1, image.size(1) - width + 1)
    for row in range(output.size(0)):
        for col in range(output.size(1)):
            output[row, col] = (image[row:row + height, col:col + width] * kernel).sum()
    return output


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, images):
        if images.ndim != 4 or images.size(1) != 1:
            raise ValueError("expected grayscale images of shape (B, 1, H, W)")
        return self.classifier(self.features(images).flatten(1))


def make_stripes(batch=32, size=16):
    """标签 0：横线；标签 1：竖线。每批重新加入轻微噪声。"""
    labels = torch.arange(batch) % 2
    images = 0.1 * torch.randn(batch, 1, size, size)
    for index, label in enumerate(labels):
        position = int(torch.randint(3, size - 3, ()).item())
        if label == 0:
            images[index, 0, position - 1:position + 2, :] += 1
        else:
            images[index, 0, :, position - 1:position + 2] += 1
    return images, labels


def demo(steps=12):
    torch.manual_seed(42)
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        images, labels = make_stripes()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"CNN logits={tuple(logits.shape)} loss={loss.item():.4f}")
    print("corr2d:", corr2d(torch.ones(4, 4), torch.ones(2, 2)).shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="运行不下载数据的短实验")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    demo(args.steps)
