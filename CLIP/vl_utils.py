from pathlib import Path
import torch

CLIP_ID = 'openai/clip-vit-base-patch32'
LABELS_EN = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']
LABELS_ZH = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']


def get_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def load_model(model_id, device):
    """预训练模型很大，第一次调用时由 Hugging Face 下载权重。"""
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model, processor


def score_pairs(model, processor, images, texts, model_id, device):
    """输出图文匹配 Logits: (num_texts, num_images)。

    使用 forward().logits_per_text，避免 transformers 不同版本的
    get_image_features / get_text_features 返回对象类型存在差异。
    """
    kwargs = dict(text=texts, images=images, return_tensors='pt')
    kwargs.update(padding=True, truncation=True)
    inputs = processor(**kwargs).to(device)
    with torch.inference_mode():
        scores = model(**inputs).logits_per_text
    return scores.float().cpu()  # (num_texts, num_images)


def get_cifar10_subset(root, per_class=10, seed=42):
    """从 CIFAR-10 测试集随机按类别均衡采样（不进行任何微调）。"""
    from torchvision import datasets
    ds = datasets.CIFAR10(root=str(root), train=False, download=True)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=generator).tolist()
    cnt = [0] * 10
    samples = []
    for i in indices:
        image, target = ds[i]
        if cnt[target] < per_class:
            samples.append((image.convert('RGB'), target))
            cnt[target] += 1
        if all(x >= per_class for x in cnt):
            break
    if not all(x == per_class for x in cnt):
        raise ValueError('样本不够，请调小 per_class')
    return samples


def zero_shot(model_id, lang, samples, batch_size, device):
    """对于候选文本的 Logits 做 argmax；Softmax 只用于同一模型内部排序。"""
    model, processor = load_model(model_id, device)
    labels = LABELS_ZH if lang == 'zh' else LABELS_EN
    texts = [f'一张{c}的照片。' for c in labels] if lang == 'zh' else [f'a photo of a {c}.' for c in labels]
    preds = []
    scores_all = []
    for start in range(0, len(samples), batch_size):
        images = [p[0] for p in samples[start:start + batch_size]]
        scores = score_pairs(model, processor, images, texts, model_id, device).T  # (B, 10)
        preds.extend(scores.argmax(dim=-1).tolist())
        scores_all.append(scores)
    return preds, torch.cat(scores_all, dim=0), labels


def plot_examples(samples, preds, labels, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N = min(12, len(samples))
    fig, axs = plt.subplots((N + 3) // 4, 4, figsize=(11, 3 * ((N + 3) // 4)), squeeze=False)
    for ax in axs.flat:
        ax.axis('off')
    for i, (image, label) in enumerate(samples[:N]):
        ax = axs[i // 4, i % 4]
        ax.imshow(image)
        # 图片标题只输出数字序号，避免无中文字库时 Matplotlib 缺字
        ax.set_title(f'true={label}, pred={preds[i]}')
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)
