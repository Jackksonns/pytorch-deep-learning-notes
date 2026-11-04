"""加载 DDPM 权重，从纯高斯噪声生成 MNIST 图片并展示采样中间步骤。"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from DDPM import DDPM, DDPMConfig, SimpleUNet

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=HERE / 'results' / 'ddpm_mnist.pth')
    p.add_argument('--num-images', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.num_images < 1:
        p.error('num-images 必须大于 0')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = DDPMConfig(**checkpoint['config'])
    diffusion = DDPM(SimpleUNet(config.base_channels), config.timesteps, config.schedule).to(device)
    diffusion.model.load_state_dict(checkpoint['model'])
    diffusion.eval(); torch.manual_seed(args.seed)
    T = config.timesteps
    steps = sorted(set([T - 1, 3 * T // 4, T // 2, T // 4, 0]), reverse=True)
    generated, traces = diffusion.sample(args.num_images, device, capture_steps=steps)
    result = HERE / 'results'; result.mkdir(exist_ok=True)
    tag = '_demo' if checkpoint.get('demo') else ''
    save_image((generated.clamp(-1, 1) + 1) / 2, result / f'sampled_digits{tag}.png', nrow=4)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.5 * len(steps), 3))
    for ax, step in zip(axes, steps):
        ax.imshow(((traces[step][0, 0].clamp(-1, 1) + 1) / 2).numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f't={step}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(result / f'reverse_process{tag}.png', dpi=160); plt.close(fig)
    print('图像已保存到:', result)
    if checkpoint.get('demo'):
        print('注意：当前权重来自 FakeData demo，无法生成有意义的手写数字')


if __name__ == '__main__':
    main()
