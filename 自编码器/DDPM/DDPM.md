# DDPM：基于 MNIST 的扩散生成模型

## 1. 模型简介

 `自编码器/VAE/` 使用潜变量完成生成任务。DDPM 则从另一条路线生成图像：先学习图像如何逐渐被噪声扰动，再训练网络预测每一步的噪声，最后从随机高斯噪声逐步恢复图像。

## 2. 核心原理

定义 `beta_t` 为当前步噪声方差，`alpha_t=1-beta_t`，`alpha_bar_t=∏_{s=1}^t alpha_s`。为方便代码实现，时间索引使用 `t=0,1,...,T-1`，`alpha_bar[0]=alpha_0`。

**前向加噪（闭式采样）**：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
其中 epsilon ~ N(0, I)
```

**训练网络预测 epsilon**：

```text
loss = MSE(epsilon_theta(x_t, t), epsilon)
```

**反向采样**：先利用预测的 epsilon 估计 x0，再使用后验 `q(x_{t-1} | x_t, x0_pred)` 的均值与方差得到上一个时间步的图像。`t=0` 时不额外加入随机噪声。

本实验使用的 U-Net 包含：卷积、残差块、Sinusoidal Time Embedding、两级下采样、Skip Connection 和上采样。

## 3. 文件说明

| 文件 | 内容 |
| --- | --- |
| `DDPM.py` | Cosine/Linear Schedule、`q_sample()`、`p_sample()`、`SimpleUNet`、`sample()` |
| `train.py` | MNIST 数据加载、随机时间步、MSE、保存训练 Loss 与模型 |
| `sample.py` | 从 checkpoint 加载权重、纯噪声生成图片、记录中间步骤 |
| `results/` | 运行后保存曲线、正向加噪图、反向采样图 |

## 4. 环境与运行

```bash
python -m pip install -r requirements.txt
cd 自编码器/DDPM

# 真实 MNIST 训练；初次需要下载数据集，建议使用 GPU
python train.py --epochs 15 --timesteps 200 --schedule cosine --batch-size 128

# 载入训练后的模型，执行完整反向采样
python sample.py --checkpoint results/ddpm_mnist.pth --num-images 16
```

### 先做离线 Smoke Test

```bash
python train.py --demo --epochs 1 --batch-size 4 --max-batches 1 --timesteps 8 --base-channels 8 --sample-count 2
python sample.py --checkpoint results/ddpm_demo.pth --num-images 2
```

`--demo` 使用 FakeData，**生成图像不代表真正数字，也不能用于评价算法效果**。

这里设置 `timesteps=200` 搭配 Cosine Schedule，有助于让最后时间步接近纯噪声。直接将经典 1000 步线性 Schedule 改成很短的步数，并不一定满足起始生成所需的近纯噪声假设。

