"""训练 MNIST DDPM；--demo 使用 FakeData，只测试流程，不代表生成质量。"""
import argparse
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from DDPM import DDPM, DDPMConfig, SimpleUNet

HERE = Path(__file__).resolve().parent


def get_dataset(args):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if args.demo:
        return datasets.FakeData(size=128, image_size=(1, 28, 28), num_classes=10, transform=tf)
    return datasets.MNIST(root=str(args.data_root), train=True, download=True, transform=tf)


def plot_noise_stages(diffusion, x0, output):
    """同一 x0、同一 epsilon 下的边际噪声图，不代表单条马尔可夫轨迹。"""
    T = diffusion.timesteps
    steps = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
    epsilon = torch.randn_like(x0)
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(2.3 * (len(steps) + 1), 2.5))
    axes[0].imshow((x0[0, 0].cpu().numpy() + 1) / 2, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('original'); axes[0].axis('off')
    for ax, t_int in zip(axes[1:], steps):
        t = torch.full((1,), t_int, dtype=torch.long, device=x0.device)
        x_t, _ = diffusion.q_sample(x0, t, epsilon)
        ax.imshow((x_t[0, 0].cpu().numpy() + 1) / 2, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f't={t_int}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action='store_true')
    p.add_argument('--data-root', type=Path, default=HERE / 'data')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--schedule', choices=['cosine', 'linear'], default='cosine')
    p.add_argument('--base-channels', type=int, default=32)
    p.add_argument('--max-batches', type=int, default=None)
    p.add_argument('--sample-count', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.timesteps < 2 or args.sample_count < 1:
        p.error('epochs、sample-count 必须大于 0，timesteps 至少为 2')
    if args.max_batches is not None and args.max_batches <= 0:
        p.error('max-batches 必须大于 0')
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    ds = get_dataset(args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    config = DDPMConfig(timesteps=args.timesteps, schedule=args.schedule, base_channels=args.base_channels)
    diffusion = DDPM(SimpleUNet(config.base_channels), config.timesteps, config.schedule).to(device)
    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=args.lr)
    result = HERE / 'results'; result.mkdir(exist_ok=True)
    tag = '_demo' if args.demo else ''
    first, _ = next(iter(loader))
    plot_noise_stages(diffusion, first[:1].to(device), result / f'noise_process{tag}.png')
    losses = []
    print(f'device={device}, params={sum(p.numel() for p in diffusion.model.parameters()):,}, demo={args.demo}')
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        total_loss = total_batches = 0
        for batch_id, (images, _) in enumerate(loader):
            if args.max_batches is not None and batch_id >= args.max_batches:
                break
            images = images.to(device)
            loss = diffusion(images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); total_batches += 1
        if total_batches == 0:
            raise RuntimeError('没有训练 Batch')
        avg_loss = total_loss / total_batches
        losses.append(avg_loss)
        print(f'epoch {epoch:02d} | mean MSE loss {avg_loss:.5f}')
    output = result / ('ddpm_demo.pth' if args.demo else 'ddpm_mnist.pth')
    torch.save({'model': diffusion.model.state_dict(), 'config': asdict(config),
                'demo': args.demo, 'seed': args.seed}, output)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, label='noise MSE loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); fig.tight_layout()
    fig.savefig(result / f'training_loss{tag}.png', dpi=160); plt.close(fig)
    diffusion.eval()
    generated, _ = diffusion.sample(args.sample_count, device)
    save_image((generated.clamp(-1, 1) + 1) / 2,
               result / f'generated_digits{tag}.png', nrow=4)
    print(f'权重、训练曲线、生成结果已保存: {result}')
    if args.demo:
        print('警告：--demo 使用随机 FakeData，图像没有数字语义，不能用来判断 DDPM 效果')


if __name__ == '__main__':
    main()
