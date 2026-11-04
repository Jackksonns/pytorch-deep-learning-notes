"""用最后一层 CLS→Patch Attention 绘图（注意权重不是严格的解释性证据）。"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ViT import ViTConfig, VisionTransformer
from train import get_datasets, MEAN, STD

HERE = Path(__file__).resolve().parent


def draw_attention(image, weights, output):
    """image: (3, H, W)，weights: (heads, tokens, tokens)。"""
    # CLS 是第 0 个 Token。取最后一层 CLS 对所有 Patch 的平均注意力。
    n = weights.size(-1) - 1
    side = int(n ** 0.5)
    if side * side != n:
        raise ValueError('当前只支持正方形 Patch 网格')
    cls_to_patch = weights[:, 0, 1:].mean(dim=0).view(1, 1, side, side)
    heatmap = F.interpolate(cls_to_patch, size=image.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    image = (image * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1))
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(image); axes[0].set_title('Input image'); axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].imshow(heatmap.numpy(), cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title('CLS attention (last layer)'); axes[1].axis('off')
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=HERE / 'results' / 'vit_p4.pth')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--data-root', type=Path, default=HERE / 'data')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if ckpt['arch'] != 'vit':
        p.error('该工具只支持 ViT checkpoint')
    net = VisionTransformer(ViTConfig(**ckpt['config'])).to(device)
    net.load_state_dict(ckpt['model']); net.eval()
    _, _, ds = get_datasets(ckpt.get('demo', False), ckpt['seed'], args.data_root)
    if args.index < 0 or args.index >= len(ds):
        p.error('index 超出测试数据集范围')
    image, label = ds[args.index]
    with torch.no_grad():
        logits, weights_list = net(image.unsqueeze(0).to(device), return_attention=True)
    out = HERE / 'results' / ('vit_attention_demo.png' if ckpt.get('demo') else 'vit_attention.png')
    draw_attention(image.cpu(), weights_list[-1][0].cpu(), out)
    print(f'label={label}, pred={logits.argmax(dim=-1).item()}, attention 图已保存: {out}')
    if ckpt.get('demo'):
        print('注意：FakeData + 随机训练的注意力图没有语义解释价值。')


if __name__ == '__main__':
    main()
