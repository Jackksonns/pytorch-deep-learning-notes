"""从零实现小型 Decoder-only Transformer（字符级 Mini-GPT）。"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """因果自注意力：位置 t 只能访问 0~t 的 Token。"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, '隐藏维度必须被注意力头数整除'
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 注册为 Buffer，保存模型时不必将固定 Mask 写进权重
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer('causal_mask', mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, X):
        B, T, C = X.shape  # (batch_size, seq_len, n_embd)
        q, k, v = self.qkv(X).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float('-inf'))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        Y = weights @ v
        Y = Y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(Y))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block：LayerNorm + Attention/MLP + 残差。"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, X):
        X = X + self.attn(self.ln1(X))
        X = X + self.mlp(self.ln2(X))
        return X


class GPT(nn.Module):
    """输入 Token ID，预测每个位置的下一个 Token。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # 输入 Embedding 和输出投影共用参数，减少模型参数量
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T)，targets: (B, T)，每个位置的目标是下一个字符
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f'序列长度 {T} 超过 block_size={self.config.block_size}')
        pos = torch.arange(T, device=idx.device)
        X = self.drop(self.token_embedding(idx) + self.position_embedding(pos)[None, :, :])
        X = self.ln_f(self.blocks(X))
        logits = self.lm_head(X)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        """逐字符采样；超过上下文窗口时仅保留最近的 block_size 个 Token。"""
        if temperature <= 0:
            raise ValueError('temperature 必须大于 0')
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError('top_k 必须大于 0')
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


if __name__ == '__main__':
    config = GPTConfig(vocab_size=30, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(tokens, tokens)
    print('logits:', logits.shape, 'loss:', loss.item())
    print('generated:', model.generate(tokens[:1], max_new_tokens=3).shape)
