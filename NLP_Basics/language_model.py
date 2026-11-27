"""字符级下一词元预测：手写 RNN、LSTM 与 GRU。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from rnn_lstm import ScratchRecurrent

# 进一步学习可接入 WikiText-2：https://huggingface.co/datasets/Salesforce/wikitext
TEXT = "the cat sat on the mat. the dog ran in the sun. " * 12


class CharacterLM(nn.Module):
    def __init__(self, vocab_size, cell, embed_dim=24, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.recurrent = ScratchRecurrent(embed_dim, hidden_size, cell)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, state=None):
        outputs, state = self.recurrent(self.embedding(tokens), state)
        return self.head(outputs), state  # (B, T, V)


def demo(cell="rnn", steps=12):
    torch.manual_seed(42)
    chars = sorted(set(TEXT))
    to_id = {char: i for i, char in enumerate(chars)}
    ids = torch.tensor([to_id[char] for char in TEXT])
    model = CharacterLM(len(chars), cell)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - 17, (8,))
        x = torch.stack([ids[i:i + 16] for i in starts])
        y = torch.stack([ids[i + 1:i + 17] for i in starts])  # 只预测下一字符
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    generated = [to_id["t"]]
    state = None
    with torch.no_grad():
        for _ in range(20):
            scores, state = model(torch.tensor([[generated[-1]]]), state)
            generated.append(int(scores[0, -1].argmax()))
    print(f"{cell} logits={tuple(logits.shape)} loss={loss.item():.4f}")
    print("sample:", "".join(chars[i] for i in generated))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="运行内置短文本实验")
    parser.add_argument("--cell", choices=["rnn", "lstm", "gru"], default="rnn")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    demo(args.cell, args.steps)
