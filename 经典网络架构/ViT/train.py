"""在 CIFAR-10 上训练 ViT，也可用同一数据划分训练 ResNet18 作基线。"""
import argparse
import random
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from ViT import VisionTransformer, ViTConfig

HERE = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_datasets(demo, seed, data_root):
    """训练集与验证集使用相同索引划分，不同增强；测试集始终独立。"""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if demo:
        # FakeData 生成随机图片与标签，只用于测试脚本，准确率不代表真实模型性能
        train_full = datasets.FakeData(size=200, image_size=(3, 32, 32), num_classes=10,
                                       transform=train_tf, random_offset=10)
        eval_full = datasets.FakeData(size=200, image_size=(3, 32, 32), num_classes=10,
                                      transform=eval_tf, random_offset=10)
        test = datasets.FakeData(size=40, image_size=(3, 32, 32), num_classes=10,
                                 transform=eval_tf, random_offset=1000)
    else:
        train_full = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_tf)
        eval_full = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=eval_tf)
        test = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=eval_tf)
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    split = max(1, int(0.1 * len(idx)))
    return Subset(train_full, idx[split:]), Subset(eval_full, idx[:split]), test


def build_model(args):
    if args.arch == 'resnet18':
        net = models.resnet18(weights=None)
        # CIFAR-10 图像只有 32x32，不使用 ImageNet 的 7x7 卷积和 MaxPool
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(net.fc.in_features, 10)
        return net, None
    config = ViTConfig(img_size=32, patch_size=args.patch_size,
                       embed_dim=args.embed_dim, num_heads=args.heads,
                       num_layers=args.layers, dropout=args.dropout)
    return VisionTransformer(config), asdict(config)


def run_epoch(net, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    net.train(training)
    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(training):
        for batch_i, (X, y) in enumerate(loader):
            if max_batches is not None and batch_i >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            logits = net(X)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)
    if total == 0:
        raise ValueError('没有可用的 Batch，请检查数据集与 --max-batches')
    return total_loss / total, total_correct / total


def plot_predictions(net, loader, device, output, max_items=8):
    net.eval()
    X, y = next(iter(loader))
    X, y = X[:max_items].to(device), y[:max_items]
    with torch.no_grad():
        pred = net(X).argmax(dim=-1).cpu()
    X = X.cpu() * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)
    cols = 4
    rows = (len(X) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9, rows * 2.7), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i in range(len(X)):
        axes[i // cols, i % cols].imshow(X[i].permute(1, 2, 0).clamp(0, 1))
        axes[i // cols, i % cols].set_title(f'true={int(y[i])} / pred={int(pred[i])}')
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['vit', 'resnet18'], default='vit')
    p.add_argument('--demo', action='store_true', help='FakeData，仅检查流程')
    p.add_argument('--data-root', type=Path, default=HERE / 'data')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--patch-size', type=int, default=4)
    p.add_argument('--embed-dim', type=int, default=192)
    p.add_argument('--heads', type=int, default=3)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max-batches', type=int, default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs <= 0 or (args.max_batches is not None and args.max_batches <= 0):
        p.error('epochs 和 max-batches 必须大于 0')
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    train_ds, val_ds, test_ds = get_datasets(args.demo, args.seed, args.data_root)
    train_iter = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_iter = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_iter = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    net, config = build_model(args)
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    history = []
    print(f'arch={args.arch}, device={device}, params={sum(x.numel() for x in net.parameters()):,}, demo={args.demo}')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(net, train_iter, criterion, device, optimizer, args.max_batches)
        val_loss, val_acc = run_epoch(net, val_iter, criterion, device, max_batches=args.max_batches)
        history.append((epoch, train_loss, val_loss, train_acc, val_acc))
        print(f'epoch {epoch:2d} | train loss {train_loss:.4f}, acc {train_acc:.3f}'
              f' | val loss {val_loss:.4f}, acc {val_acc:.3f}')
    test_loss, test_acc = run_epoch(net, test_iter, criterion, device, max_batches=args.max_batches)
    print(f'test loss {test_loss:.4f} | test acc {test_acc:.3f}')
    result = HERE / 'results'; result.mkdir(exist_ok=True)
    suffix = '_demo' if args.demo else ''
    name = f'{args.arch}_p{args.patch_size}{suffix}' if args.arch == 'vit' else f'{args.arch}{suffix}'
    torch.save({'model': net.state_dict(), 'config': config, 'arch': args.arch,
                'seed': args.seed, 'test_acc': test_acc, 'demo': args.demo}, result / f'{name}.pth')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x[0] for x in history], [x[1] for x in history], label='train loss')
    ax.plot([x[0] for x in history], [x[2] for x in history], label='val loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); fig.tight_layout()
    fig.savefig(result / f'{name}_loss.png', dpi=160); plt.close(fig)
    plot_predictions(net, test_iter, device, result / f'{name}_predictions.png')
    print(f'已保存到 {result}（若 --demo，图像和准确率不代表实际表现）')


if __name__ == '__main__':
    main()
