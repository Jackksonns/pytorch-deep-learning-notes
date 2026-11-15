"""BiLSTM 序列标注：每个输入词对应一个标签。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

# 进一步学习可接入 CoNLL-2003：https://huggingface.co/datasets/eriktks/conll2003
# 这里的标签仅用于演示
EXAMPLES = [
    (["alice", "visits", "paris"], [1, 0, 2]),
    (["bob", "likes", "london"], [1, 0, 2]),
    (["alice", "likes", "london", "today"], [1, 0, 2, 0]),
    (["bob", "visits", "paris", "today"], [1, 0, 2, 0]),
]
TAGS = ["O", "PER", "LOC"]


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim=24, hidden_size=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True,
                            bidirectional=True, num_layers=2)
        self.head = nn.Linear(2 * hidden_size, len(TAGS))

    def forward(self, tokens):
        lengths = (tokens != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(tokens), lengths,
                                                     batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True,
                                                     total_length=tokens.size(1))
        return self.head(output)  # (B, T, num_tags)


def demo(steps=15):
    torch.manual_seed(42)
    vocab = {word: i + 1 for i, word in enumerate(sorted({w for words, _ in EXAMPLES for w in words}))}
    width = max(len(words) for words, _ in EXAMPLES)
    x = torch.zeros(len(EXAMPLES), width, dtype=torch.long)
    y = torch.full((len(EXAMPLES), width), -100, dtype=torch.long)
    for i, (words, labels) in enumerate(EXAMPLES):
        x[i, :len(words)] = torch.tensor([vocab[w] for w in words])
        y[i, :len(labels)] = torch.tensor(labels)
    model = BiLSTMTagger(len(vocab) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(TAGS)), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"BiLSTM tags={tuple(logits.shape)} loss={loss.item():.4f} ignored={(y == -100).sum().item()}")
    print("first sentence:", [TAGS[i] for i in model(x)[0, :3].argmax(-1).tolist()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="运行不下载数据的短实验（默认行为）")
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    demo(args.steps)
