"""CLIP 零样本 CIFAR-10 分类：不训练模型，直接比较图文相似度。"""
import argparse
from pathlib import Path
from vl_utils import (CLIP_ID, get_device, get_cifar10_subset,
                      zero_shot, plot_examples)

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=CLIP_ID)
    p.add_argument('--language', choices=['en', 'zh'], default='en')
    p.add_argument('--per-class', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-root', type=Path, default=HERE / 'data')
    args = p.parse_args()
    if args.per_class < 1 or args.batch_size < 1:
        p.error('--per-class 与 --batch-size 必须大于 0')
    samples = get_cifar10_subset(args.data_root, args.per_class, args.seed)
    predictions, logits, labels = zero_shot(args.model, args.language, samples,
                                             args.batch_size, get_device(args.device))
    targets = [y for _, y in samples]
    correct = sum(int(y == p) for y, p in zip(targets, predictions))
    print(f'Model: {args.model} | Language: {args.language} | Total: {len(samples)}')
    print(f'Zero-shot Top-1 Accuracy: {correct / len(samples):.4f}')
    for i in range(min(len(samples), 10)):
        print(f'[{i:02d}] true={labels[targets[i]]} pred={labels[predictions[i]]}')
    output = HERE / 'results' / f'clip_{args.language}_zero_shot.png'
    plot_examples(samples, predictions, labels, output)
    print(f'示例预测图片已保存: {output}')
    print('注意：分数只在当前模型与候选标签集合中解释，不能跨模型视为统一概率。')


if __name__ == '__main__':
    main()
