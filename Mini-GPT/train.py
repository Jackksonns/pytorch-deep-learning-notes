"""Mini-GPT 训练：字符级 Tokenizer + 随机连续文本片段。"""
import argparse
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlretrieve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from GPT import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def get_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_text(path, download=False, demo=False):
    """正常训练需提供真实文本；--demo 只用于离线验证，不能评价生成质量。"""
    if demo:
        return ('To be, or not to be: that is the question.\n'
                'Learning a small language model from characters.\n') * 40
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if download:
            print('正在下载 Tiny Shakespeare ...')
            urlretrieve(SHAKESPEARE_URL, path)
        else:
            raise FileNotFoundError(f'未找到 {path}，可添加 --download 下载公开数据集')
    text = path.read_text(encoding='utf-8')
    if len(text) < 100:
        raise ValueError('文本太短，无法训练；请提供至少 100 个字符')
    return text


def get_batch(data, batch_size, block_size, device):
    """随机取出连续片段，y 相对 x 整体向右移动一位。"""
    # 需要至少 block_size+1 个字符，才能生成最后一位的 target
    if len(data) <= block_size:
        raise ValueError('数据长度需要大于 block_size')
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, train_ids, val_ids, args, device):
    model.eval()
    out = {}
    for name, data in [('train', train_ids), ('val', val_ids)]:
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description='训练小型字符级 GPT')
    p.add_argument('--text', type=Path, default=HERE / 'data' / 'input.txt')
    p.add_argument('--download', action='store_true', help='下载 Tiny Shakespeare')
    p.add_argument('--demo', action='store_true', help='离线极小语料，仅用于检查代码')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--embed-dim', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval-interval', type=int, default=100)
    p.add_argument('--eval-iters', type=int, default=10)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.steps < 1 or args.eval_interval < 1 or args.eval_iters < 1:
        p.error('steps、eval-interval、eval-iters 均需要大于 0')
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    text = load_text(args.text, download=args.download, demo=args.demo)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    if min(len(train_ids), len(val_ids)) <= args.block_size:
        p.error('训练集或验证集长度不足，请调小 --block-size 或使用更长的文本')

    config = GPTConfig(len(chars), args.block_size, args.embed_dim,
                       args.heads, args.layers, args.dropout)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    print(f'device={device}, vocab_size={len(chars)}, params={sum(x.numel() for x in model.parameters()):,}')
    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_ids, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(model, train_ids, val_ids, args, device)
            print(f'step {step:5d} | train loss {metrics["train"]:.4f} | val loss {metrics["val"]:.4f}')
            history.append((step, metrics['train'], metrics['val']))

    results = HERE / 'results'
    results.mkdir(exist_ok=True)
    ckpt = {'model': model.state_dict(), 'config': asdict(config),
            'chars': chars, 'step': args.steps, 'demo': args.demo}
    tag = '_demo' if args.demo else ''
    torch.save(ckpt, results / f'mini_gpt{tag}.pth')
    plt.figure(figsize=(7, 4))
    for j, name in [(1, 'train'), (2, 'validation')]:
        plt.plot([row[0] for row in history], [row[j] for row in history], label=name)
    plt.xlabel('Step'); plt.ylabel('Cross-entropy loss'); plt.legend(); plt.tight_layout()
    plt.savefig(results / f'training_loss{tag}.png', dpi=160); plt.close()
    print(f'模型和训练曲线已保存到 {results}')


if __name__ == '__main__':
    main()
