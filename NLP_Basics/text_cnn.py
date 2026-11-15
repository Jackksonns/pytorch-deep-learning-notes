"""TextCNN：沿词序列做一维卷积"""

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=32, channels=32, kernel_sizes=(2, 3, 4)):
        super().__init__()
        self.min_length = max(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, channels, k) for k in kernel_sizes)
        self.classifier = nn.Linear(channels * len(kernel_sizes), num_classes)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.size(1) < self.min_length:
            raise ValueError(f"tokens must have shape (B, T) with T >= {self.min_length}")
        x = self.embedding(tokens).transpose(1, 2)  # (B, E, T)
        features = [F.relu(conv(x)).amax(dim=-1) for conv in self.convs]
        return self.classifier(torch.cat(features, dim=-1))


if __name__ == "__main__":
    torch.manual_seed(42)
    model = TextCNN(vocab_size=20, num_classes=2)
    scores = model(torch.randint(1, 20, (3, 6)))
    scores.sum().backward()
    print("TextCNN logits:", scores.shape)
