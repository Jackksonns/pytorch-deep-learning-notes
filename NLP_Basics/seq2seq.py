"""LSTM Encoder–Decoder：教师强制训练与逐步贪心生成。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

# 进一步学习可接入 OPUS Books：https://huggingface.co/datasets/Helsinki-NLP/opus_books
PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE = 13  # 3..12 表示十个数字


class Seq2Seq(nn.Module):
    def __init__(self, embed_dim=24, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.encoder = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, VOCAB_SIZE)

    def encode(self, source):
        _, state = self.encoder(self.embedding(source))
        return state

    def forward(self, source, decoder_input):
        output, _ = self.decoder(self.embedding(decoder_input), self.encode(source))
        return self.head(output)  # (B, target_steps, vocab_size)

    def generate(self, source, max_steps=8):
        state = self.encode(source)
        token = torch.full((source.size(0), 1), BOS, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        predictions = []
        for _ in range(max_steps):
            output, state = self.decoder(self.embedding(token), state)
            token = self.head(output).argmax(-1)
            token = torch.where(finished[:, None], EOS, token)
            predictions.append(token)
            finished |= token[:, 0].eq(EOS)
            if finished.all():
                break
        return torch.cat(predictions, dim=1)


def demo(steps=20, task="reverse"):
    torch.manual_seed(42)
    model = Seq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        digits = torch.randint(3, VOCAB_SIZE, (16, 4))
        source = torch.cat((digits, torch.full((16, 1), EOS)), dim=1)
        target_digits = digits.flip(1) if task == "reverse" else digits
        target = torch.cat((target_digits, torch.full((16, 1), EOS)), dim=1)
        decoder_input = torch.cat((torch.full((16, 1), BOS), target[:, :-1]), dim=1)
        logits = model(source, decoder_input)  # 教师强制：输入真实的上一词元
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sample = torch.tensor([[3, 4, 5, 6, EOS]])
    model.eval()
    with torch.no_grad():
        prediction = model.generate(sample)
    print(f"seq2seq {task} logits={tuple(logits.shape)} loss={loss.item():.4f}")
    print("source:", sample[0].tolist(), "generated:", prediction[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="运行不下载数据的短实验（默认行为）")
    parser.add_argument("--task", choices=["copy", "reverse"], default="reverse")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    demo(args.steps, args.task)
