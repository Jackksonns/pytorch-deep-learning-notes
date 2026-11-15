"""从零实现单层 RNN/LSTM；输入与输出均采用 batch-first。"""

import torch
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        # x: (B, D), h: (B, H)
        return torch.tanh(self.input_linear(x) + self.hidden_linear(h))


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        forget, input_gate, output, candidate = (
            self.input_linear(x) + self.hidden_linear(h)
        ).chunk(4, dim=-1)
        forget = forget.sigmoid()
        input_gate = input_gate.sigmoid()
        output = output.sigmoid()
        candidate = candidate.tanh()
        c = forget * c + input_gate * candidate
        h = output * c.tanh()
        return h, c


class ScratchRecurrent(nn.Module):
    """沿时间维展开一个共享参数的 RNN 或 LSTM 单元。"""

    def __init__(self, input_size, hidden_size, cell="rnn"):
        super().__init__()
        if cell not in {"rnn", "lstm"}:
            raise ValueError("cell must be 'rnn' or 'lstm'")
        self.cell_type = cell
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size) if cell == "rnn" else LSTMCell(input_size, hidden_size)

    def forward(self, x, state=None):
        if x.ndim != 3:
            raise ValueError("expected (batch, time, features)")
        batch, steps, _ = x.shape
        if steps == 0:
            raise ValueError("sequence length must be positive")
        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            state = h if self.cell_type == "rnn" else (h, h.clone())
        outputs = []
        for t in range(steps):
            state = self.cell(x[:, t], state)
            outputs.append(state if self.cell_type == "rnn" else state[0])
        return torch.stack(outputs, dim=1), state


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)
    for cell in ("rnn", "lstm"):
        model = ScratchRecurrent(8, 12, cell)
        y, _ = model(x)
        y.sum().backward()
        print(cell, y.shape, model.cell.input_linear.weight.grad.shape)
