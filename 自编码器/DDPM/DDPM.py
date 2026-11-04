"""DDPM：手写正向加噪、时间条件 U-Net、反向去噪采样。"""
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DDPMConfig:
    timesteps: int = 200
    schedule: str = 'cosine'
    base_channels: int = 32
    image_size: int = 28


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine Noise Schedule，保证终点接近纯噪声。"""
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1e-6, max=0.999).float()


def linear_beta_schedule(timesteps):
    """经典 1000 步线性调度；短步数应优先选择 Cosine。"""
    return torch.linspace(1e-4, 0.02, timesteps)


def extract(values, t, shape):
    """根据时间步 t 取系数，并扩展为 (B, 1, 1, 1) 用于图像广播。"""
    return values.gather(0, t).view(t.shape[0], *((1,) * (len(shape) - 1)))


class SinusoidalPositionEmbedding(nn.Module):
    """将时间步 t 编码为 sin/cos 向量。"""
    def __init__(self, dim):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError('时间编码维度需要是 >=4 的偶数')
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        angles = t.float()[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """卷积 ResBlock：把时间嵌入投影后加到中间特征。"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, X, time_emb):
        H = self.conv1(X)
        H = H + self.time_proj(time_emb)[:, :, None, None]
        H = self.conv2(H)
        return H + self.skip(X)


class SimpleUNet(nn.Module):
    """28x28 MNIST 小型 U-Net；输出每个像素的预测噪声。"""
    def __init__(self, base_channels=32):
        super().__init__()
        if base_channels % 8:
            raise ValueError('base_channels 应为 8 的倍数')
        c = base_channels
        t_dim = c * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbedding(c), nn.Linear(c, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(1, c, 3, padding=1)
        self.enc1 = ResidualBlock(c, c, t_dim)
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)       # 28 -> 14
        self.enc2 = ResidualBlock(c * 2, c * 2, t_dim)
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)   # 14 -> 7
        self.mid = ResidualBlock(c * 4, c * 4, t_dim)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(c * 4, c * 2, t_dim)                  # concat 14x14 skip
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(c * 2, c, t_dim)                      # concat 28x28 skip
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, X, t):
        if X.shape[-2:] != (28, 28) or X.shape[1] != 1:
            raise ValueError('本教学 U-Net 只支持 [B, 1, 28, 28]')
        time_emb = self.time_mlp(t)
        X = self.in_conv(X)
        skip1 = self.enc1(X, time_emb)    # (B, c, 28, 28)
        skip2 = self.enc2(self.down1(skip1), time_emb)  # (B, 2c, 14, 14)
        H = self.mid(self.down2(skip2), time_emb)        # (B, 4c, 7, 7)
        H = self.dec2(torch.cat([self.up2(H), skip2], dim=1), time_emb)
        H = self.dec1(torch.cat([self.up1(H), skip1], dim=1), time_emb)
        return self.out(H)              # (B, 1, 28, 28)


class DDPM(nn.Module):
    """DDPM 调度与采样封装；训练目标为预测加噪时使用的 epsilon。"""
    def __init__(self, model, timesteps=200, schedule='cosine'):
        super().__init__()
        if timesteps < 2:
            raise ValueError('timesteps 至少为 2')
        self.model = model
        self.timesteps = timesteps
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f'未知 Schedule: {schedule}')
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bar', (1 - alpha_bar).sqrt())
        self.register_buffer('posterior_variance', betas * (1 - alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer('posterior_coef1', betas * alpha_bar_prev.sqrt() / (1 - alpha_bar))
        self.register_buffer('posterior_coef2', alphas.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar))

    def q_sample(self, x0, t, noise=None):
        """q(x_t | x_0) 的闭式采样，无需循环加噪。"""
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = extract(self.sqrt_alpha_bar, t, x0.shape) * x0 + \
              extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        return x_t, noise

    def forward(self, x0):
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """使用 epsilon 预测得到 x0，再根据 q(x_{t-1}|x_t,x0) 采样。"""
        pred_noise = self.model(x_t, t)
        abar = extract(self.alpha_bar, t, x_t.shape)
        x0_pred = (x_t - (1 - abar).sqrt() * pred_noise) / abar.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)  # 对 MNIST 的归一化范围进行截断
        mean = extract(self.posterior_coef1, t, x_t.shape) * x0_pred + \
               extract(self.posterior_coef2, t, x_t.shape) * x_t
        variance = extract(self.posterior_variance, t, x_t.shape).clamp(min=0)
        # t=0 时后验方差为 0，不再向最终生成图片增加噪声
        mask = (t > 0).float().view(-1, *((1,) * (x_t.ndim - 1)))
        return mean + mask * variance.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, batch_size, device, shape=(1, 28, 28), capture_steps=()):
        """从 N(0, I) 开始反向采样，可保存中间结果。"""
        self.eval()
        x = torch.randn((batch_size, *shape), device=device)
        traces = {}
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
            if i in capture_steps:
                traces[i] = x.detach().cpu().clone()
        return x, traces


if __name__ == '__main__':
    net = SimpleUNet(base_channels=8)
    diffusion = DDPM(net, timesteps=16)
    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    loss = diffusion(x0)
    loss.backward()
    print('loss:', loss.item())
    sample, _ = diffusion.sample(2, 'cpu')
    print('generated shape:', sample.shape)
