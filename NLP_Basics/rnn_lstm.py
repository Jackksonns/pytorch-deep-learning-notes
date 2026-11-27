"""从零实现单层 RNN、LSTM、GRU；输入与输出均采用 batch-first。"""

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


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x, h):
        # x: (B, D), h: (B, H)
        x_reset, x_update, x_candidate = self.input_linear(x).chunk(3, dim=-1)
        h_reset, h_update, h_candidate = self.hidden_linear(h).chunk(3, dim=-1)
        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        candidate = torch.tanh(x_candidate + reset * h_candidate)
        return (1 - update) * candidate + update * h


class ScratchRecurrent(nn.Module):
    """沿时间维展开一个共享参数的 RNN、LSTM 或 GRU 单元。"""

    def __init__(self, input_size, hidden_size, cell="rnn"):
        super().__init__()
        if cell not in {"rnn", "lstm", "gru"}:
            raise ValueError("cell must be 'rnn', 'lstm' or 'gru'")
        self.cell_type = cell
        self.hidden_size = hidden_size
        cells = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}
        self.cell = cells[cell](input_size, hidden_size)

    def forward(self, x, state=None):
        if x.ndim != 3:
            raise ValueError("expected (batch, time, features)")
        batch, steps, _ = x.shape
        if steps == 0:
            raise ValueError("sequence length must be positive")
        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            state = (h, h.clone()) if self.cell_type == "lstm" else h
        outputs = []
        for t in range(steps):
            state = self.cell(x[:, t], state)
            outputs.append(state[0] if self.cell_type == "lstm" else state)
        return torch.stack(outputs, dim=1), state


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)
    for cell in ("rnn", "lstm", "gru"):
        model = ScratchRecurrent(8, 12, cell)
        y, _ = model(x)
        y.sum().backward()
        print(cell, y.shape, model.cell.input_linear.weight.grad.shape)
