"""从零实现 Vision Transformer（ViT），以 CIFAR-10 的 32x32 图像为例。"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 192
    num_heads: int = 3
    num_layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1


class PatchEmbedding(nn.Module):
    """将图像划分为不重叠的 Patch，再映射到 Token 向量。"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError('图像边长必须能被 Patch Size 整除')
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        # kernel_size=stride=patch_size => 非重叠分块并完成线性投影
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # X: (B, C, H, W)
        if X.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f'期望图像尺寸 {self.img_size}x{self.img_size}，实际 {X.shape[-2:]}')
        X = self.proj(X)  # (B, embed_dim, H/P, W/P)
        return X.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


class ViTSelfAttention(nn.Module):
    """ViT 的多头自注意力，不需要因果 Mask。"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim 必须被 num_heads 整除')
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, X, return_attention=False):
        B, N, C = X.shape
        q, k, v = self.qkv(X).chunk(3, dim=-1)
        def split_heads(tensor):
            return tensor.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))
        weights = F.softmax((q @ k.transpose(-2, -1)) / self.head_dim**0.5, dim=-1)
        # 用未执行 Dropout 的注意力权重作可视化，比较直观
        attn_output = self.attn_drop(weights) @ v
        Y = attn_output.transpose(1, 2).contiguous().reshape(B, N, C)
        Y = self.proj_drop(self.proj(Y))
        return (Y, weights) if return_attention else Y


class EncoderBlock(nn.Module):
    """Pre-LN Encoder Block：Attention + MLP + 两次残差连接。"""
    def __init__(self, config):
        super().__init__()
        D = config.embed_dim
        self.ln1 = nn.LayerNorm(D)
        self.attn = ViTSelfAttention(D, config.num_heads, config.dropout)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, D * config.mlp_ratio), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D * config.mlp_ratio, D), nn.Dropout(config.dropout)
        )

    def forward(self, X, return_attention=False):
        if return_attention:
            attn_output, weights = self.attn(self.ln1(X), return_attention=True)
        else:
            attn_output = self.attn(self.ln1(X))
        X = X + attn_output
        X = X + self.mlp(self.ln2(X))
        return (X, weights) if return_attention else X


class VisionTransformer(nn.Module):
    """Patch Token + [CLS] + 位置编码 + Encoder + 分类头。"""
    def __init__(self, config=None):
        super().__init__()
        config = config or ViTConfig()
        self.config = config
        self.patch_embed = PatchEmbedding(config.img_size, config.patch_size,
                                           config.in_channels, config.embed_dim)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, X, return_attention=False):
        X = self.patch_embed(X)                    # (B, N, D)
        cls = self.cls_token.expand(X.size(0), -1, -1)  # (B, 1, D)
        X = torch.cat((cls, X), dim=1)             # (B, N+1, D)
        X = self.dropout(X + self.pos_embed)
        attns = []
        for block in self.blocks:
            if return_attention:
                X, weights = block(X, return_attention=True)
                attns.append(weights)
            else:
                X = block(X)
        logits = self.head(self.norm(X[:, 0]))     # CLS Token -> (B, num_classes)
        return (logits, attns) if return_attention else logits


if __name__ == '__main__':
    net = VisionTransformer(ViTConfig(embed_dim=48, num_heads=3, num_layers=2))
    X = torch.randn(2, 3, 32, 32)
    y, attns = net(X, return_attention=True)
    print('classification:', y.shape)
    print('attention:', attns[-1].shape)  # (2, 3, 65, 65)
