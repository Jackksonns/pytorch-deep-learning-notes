"""短句二分类：TextCNN 捕捉局部短语，BiLSTM 汇集双向上下文。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from text_cnn import TextCNN

# 进一步学习可接入 AG News：https://huggingface.co/datasets/fancyzhx/ag_news
SAMPLES = [
    ("the movie is very good", 1), ("this story is really great", 1),
    ("we like this new film", 1), ("the acting was quite nice", 1),
    ("the movie is very bad", 0), ("this story is really boring", 0),
    ("we dislike this new film", 0), ("the acting was quite poor", 0),
]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=24, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=layers,
                            batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden_size, 2)

    def forward(self, tokens):
        lengths = (tokens != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(tokens), lengths,
                                                     batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=-1))


def demo(arch="textcnn", steps=15):
    torch.manual_seed(42)
    vocab = {word: i + 1 for i, word in enumerate(sorted({w for text, _ in SAMPLES for w in text.split()}))}
    x = torch.zeros(len(SAMPLES), 6, dtype=torch.long)
    y = torch.tensor([label for _, label in SAMPLES])
    for row, (text, _) in enumerate(SAMPLES):
        ids = [vocab[word] for word in text.split()]
        x[row, :len(ids)] = torch.tensor(ids)
    model = TextCNN(len(vocab) + 1, 2) if arch == "textcnn" else BiLSTMClassifier(len(vocab) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{arch} logits={tuple(logits.shape)} loss={loss.item():.4f}")
    print("predictions:", model(x).argmax(-1).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="运行不下载数据的短实验（默认行为）")
    parser.add_argument("--model", choices=["textcnn", "bilstm"], default="textcnn")
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    demo(args.model, args.steps)
