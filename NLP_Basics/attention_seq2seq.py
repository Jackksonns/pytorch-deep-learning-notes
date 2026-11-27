"""加性注意力 LSTM Encoder–Decoder：逐步读取源序列并生成目标序列。"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from seq2seq import BOS, EOS, PAD, VOCAB_SIZE, make_demo_batch


HERE = Path(__file__).resolve().parent


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, valid_positions):
        # query: (B, H), keys: (B, S, H), valid_positions: (B, S)
        features = torch.tanh(
            self.key_projection(keys) + self.query_projection(query).unsqueeze(1)
        )
        scores = self.score(features).squeeze(-1)
        weights = F.softmax(scores.masked_fill(~valid_positions, float("-inf")), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class AttentionSeq2Seq(nn.Module):
    def __init__(self, embed_dim=24, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.encoder = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.attention = AdditiveAttention(hidden_size)
        self.decoder = nn.LSTMCell(embed_dim + hidden_size, hidden_size)
        self.head = nn.Linear(2 * hidden_size, VOCAB_SIZE)

    def encode(self, source):
        valid = source.ne(PAD)
        lengths = valid.sum(1)
        if (lengths == 0).any():
            raise ValueError("each source sequence needs at least one token")
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(source), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.encoder(packed)
        keys, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=source.size(1)
        )
        return keys, valid, (hidden[-1], cell[-1])

    def decode_step(self, token, state, keys, valid):
        hidden, cell = state
        context, weights = self.attention(hidden, keys, valid)
        hidden, cell = self.decoder(
            torch.cat((self.embedding(token), context), dim=-1), (hidden, cell)
        )
        logits = self.head(torch.cat((hidden, context), dim=-1))
        return logits, (hidden, cell), weights

    def forward(self, source, decoder_input):
        keys, valid, state = self.encode(source)
        logits, weights = [], []
        for step in range(decoder_input.size(1)):
            scores, state, attention = self.decode_step(decoder_input[:, step], state, keys, valid)
            logits.append(scores)
            weights.append(attention)
        return torch.stack(logits, dim=1), torch.stack(weights, dim=1)

    def generate(self, source, max_steps=8):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        keys, valid, state = self.encode(source)
        token = torch.full((source.size(0),), BOS, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        predictions, weights = [], []
        for _ in range(max_steps):
            scores, state, attention = self.decode_step(token, state, keys, valid)
            token = scores.argmax(-1)
            token = torch.where(finished, EOS, token)
            predictions.append(token)
            weights.append(attention)
            finished |= token.eq(EOS)
            if finished.all():
                break
        return torch.stack(predictions, dim=1), torch.stack(weights, dim=1)


def save_attention_plot(source, generated, weights, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_source = source.ne(PAD)
    matrix = weights[:, valid_source].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(5, matrix.shape[1]), max(3, matrix.shape[0] * 0.6)))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(matrix.shape[1]), [str(x) for x in source[valid_source].tolist()])
    ax.set_yticks(range(matrix.shape[0]), [str(x) for x in generated.tolist()])
    ax.set_xlabel("source token")
    ax.set_ylabel("generated token")
    fig.colorbar(image, ax=ax, label="attention weight")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def demo(steps=20, task="reverse"):
    torch.manual_seed(42)
    model = AttentionSeq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        source, target, decoder_input = make_demo_batch(task=task)
        logits, _ = model(source, decoder_input)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sample = torch.tensor([[3, 4, 5, 6, EOS]])
    model.eval()
    with torch.inference_mode():
        generated, weights = model.generate(sample)
    output = HERE / "results" / f"attention_alignment_{task}.png"
    save_attention_plot(sample[0], generated[0], weights[0], output)
    print(f"attention seq2seq {task} logits={tuple(logits.shape)} loss={loss.item():.4f}")
    print("source:", sample[0].tolist(), "generated:", generated[0].tolist())
    print("attention:", tuple(weights.shape), "plot:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="运行内置数字序列实验")
    parser.add_argument("--task", choices=["copy", "reverse"], default="reverse")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    demo(args.steps, args.task)
