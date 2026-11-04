"""文本检索图片；可选用 CSV 的图文配对计算 Recall@1 / Recall@5。"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from vl_utils import CLIP_ID, get_device, load_model, score_pairs

HERE = Path(__file__).resolve().parent
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def read_pairs(csv_file, image_dir, max_images):
    """CSV 必须有 image,caption 两列；每行一张不同的图片及对应文字。"""
    with csv_file.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'image', 'caption'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV 需要包含 image 和 caption 两列')
        rows = [(image_dir / r['image'], r['caption']) for r in reader][:max_images]
    if not rows:
        raise ValueError('CSV 中没有图文对')
    if len(set(str(r[0].resolve()) for r in rows)) != len(rows):
        raise ValueError('评估需要每行对应一张不同图片，避免重复标签不明确')
    return rows


def load_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert('RGB').copy())
    return images


def rank_images(model, processor, model_id, images, texts, batch_size, device):
    """每次处理一小批图片；结果 shape: (num_queries, num_images)。"""
    chunks = []
    for start in range(0, len(images), batch_size):
        part = images[start:start + batch_size]
        chunk = score_pairs(model, processor, part, texts, model_id, device)
        chunks.append(chunk)
    return torch.cat(chunks, dim=1)


def save_top5(images, paths, scores, output):
    top = torch.topk(scores, k=min(5, len(images))).indices.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(3 * len(top), 3.5), squeeze=False)
    for rank, idx in enumerate(top):
        ax = axes[0, rank]
        ax.imshow(images[idx]); ax.axis('off')
        ax.set_title(f'Top {rank + 1}: {paths[idx].name[:18]}', fontsize=9)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--query', default='a photo of a cat')
    parser.add_argument('--pairs', type=Path, default=None, help='图文配对 CSV：image,caption')
    parser.add_argument('--max-images', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.max_images < 1 or args.batch_size < 1:
        parser.error('max-images 和 batch-size 必须大于 0')
    if args.pairs:
        pairs = read_pairs(args.pairs, args.image_dir, args.max_images)
        paths, captions = [x[0] for x in pairs], [x[1] for x in pairs]
    else:
        paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in SUFFIXES)[:args.max_images]
        captions = None
    if not paths:
        parser.error('图片文件夹中没有找到图片')
    images = load_images(paths)
    device = get_device(args.device)
    model_id = CLIP_ID
    model, processor = load_model(model_id, device)
    query_scores = rank_images(model, processor, model_id, images, [args.query], args.batch_size, device)[0]
    top5 = torch.topk(query_scores, min(5, len(paths))).indices.tolist()
    print('Query:', args.query)
    for rank, i in enumerate(top5, 1):
        print(f'{rank}. {paths[i].name} | score={query_scores[i].item():.4f}')
    output = HERE / 'results' / 'clip_retrieval_top5.png'
    save_top5(images, paths, query_scores, output)
    print('检索结果已保存:', output)
    if captions is not None:
        # 仅当图片与说明严格一一对应时，才能用对角线计算 Recall@K
        all_scores = rank_images(model, processor, model_id, images, captions, args.batch_size, device)
        rankings = all_scores.argsort(dim=1, descending=True)
        correct = torch.arange(len(paths))[:, None]
        for k in [1, 5]:
            k = min(k, len(paths))
            recall = (rankings[:, :k] == correct).any(dim=1).float().mean().item()
            print(f'Text-to-image Recall@{k}: {recall:.4f}')


if __name__ == '__main__':
    main()
