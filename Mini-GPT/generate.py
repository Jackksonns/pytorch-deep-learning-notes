"""载入训练后的 Mini-GPT，在给定提示词后继续生成文本。"""
import argparse
from pathlib import Path
import torch
from GPT import GPT, GPTConfig

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=HERE / 'results' / 'mini_gpt.pth')
    parser.add_argument('--prompt', default=None, help='字符必须包含在训练词表中')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    chars = checkpoint['chars']
    stoi = {c: i for i, c in enumerate(chars)}
    prompt = args.prompt if args.prompt is not None else ('\n' if '\n' in stoi else chars[0])
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        parser.error(f'提示词包含训练词表中不存在的字符: {unknown!r}')
    if not prompt:
        parser.error('prompt 不能为空')
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        result = model.generate(x, max_new_tokens=args.tokens,
                                temperature=args.temperature, top_k=args.top_k)
    generated = ''.join(chars[i] for i in result[0].tolist())
    print(generated)
    suffix = '_demo' if checkpoint.get('demo') else ''
    output = HERE / 'results' / f'generated_text{suffix}.txt'
    output.parent.mkdir(exist_ok=True)
    output.write_text(generated, encoding='utf-8')
    print(f'\n文本已保存: {output}')


if __name__ == '__main__':
    main()
