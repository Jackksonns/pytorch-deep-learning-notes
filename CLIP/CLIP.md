# CLIP: zero-shot classification and image-text retrieval

## 1. 模型简介

CLIP uses an image encoder and a text encoder to compare images with natural-language descriptions.

本模块采用**核心损失函数手写 + 官方预训练模型推理**的方案：不尝试复现大规模图文预训练，减少显存和数据集门槛。

## 2. 核心原理

```text
图像 → Vision Encoder ─┐
                       ├→ 相似度矩阵 (num_texts, num_images)
文本 → Text Encoder  ──┘
```

### CLIP 的双向 InfoNCE 类损失

在一个包含 B 对图文的数据 Batch 中，正确匹配位于 `(B,B)` 相似度矩阵的对角线上。

- 图像检索文本：每行使用 CrossEntropyLoss。
- 文本检索图像：每列使用 CrossEntropyLoss。
- 两个方向取平均。

## 3. 文件说明

| 文件 | 主要内容 |
| --- | --- |
| `contrastive_loss.py` | CLIP contrastive loss and backward check |
| `vl_utils.py` | CLIP model loading and image-text scoring |
| `clip_zero_shot.py` | CLIP 在 CIFAR-10 测试集做零样本分类 |
| `image_text_retrieval.py` | 文本查询检索图片，支持带配对 CSV 的 Recall@k |
| `results/` | 运行后保存预测图与 Top-5 检索图 |

预训练模型标识：
- CLIP：`openai/clip-vit-base-patch32`

## 4. 运行流程

从根目录安装：

```bash
python -m pip install -r requirements.txt
```

从当前目录运行：

```bash
cd CLIP

# 不下载模型的基础损失函数测试
python contrastive_loss.py

# CIFAR-10 零样本分类。首次运行自动下载数据集与预训练权重
python clip_zero_shot.py --per-class 10 --language en --seed 42

```


预训练权重可能超过 1GB，需要联网和足够硬盘空间，CPU 推理也会较慢。**不要把模型权重上传 GitHub**。

### 自己准备图像做检索

```bash
python image_text_retrieval.py --image-dir ./my_images --query "a photo of a bicycle"
```

`my_images` 是用户自行创建的目录，至少包含一张 jpg/png/webp 图片，默认最多读取 32 张。

如果准备了每张图片唯一对应的文字描述，可以建立 `pairs.csv`：

```csv
image,caption
cat1.jpg,a white cat sitting on a sofa
bike1.jpg,a red bicycle on the street
bird1.jpg,a small bird on a tree
```

随后运行：

```bash
python image_text_retrieval.py --image-dir ./my_images --pairs ./pairs.csv --query "a photo of a cat"
```

程序会根据一一对应的图文配对计算 Text-to-image Recall@1 / Recall@5。**不要把带有多个正确描述的复杂数据集直接当作单一正确匹配问题。**

## 5. 实验建议与记录表

| 模型 | Prompt 语言 | 数据子集 | Top-1 Accuracy |
| --- | --- | --- | --- |
| CLIP ViT-B/32 | English | CIFAR-10 各类 10 张 | 待实际运行 |

还可以进一步比较不同提示词模板、类别间的易混淆情况，以及检索 Recall@k。不要在没有运行测试的情况下填写模型效果。

## 6. 问题与思考

1. 零样本分类为什么可以不训练新的分类头？
2. 使用不同候选类别集合，Softmax 相对分数为何会变化？
3. 中文与英文提示的效果差异是否由模型训练语料、Tokenizer 或提示模板共同决定？

