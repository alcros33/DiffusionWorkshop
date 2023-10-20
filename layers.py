import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NACBlock(nn.Sequential):
    """Normalization, Activation, Convolution"""
    def __init__(self, in_chs:int, out_chs:int, act_fn, norm_groups=32):
        super().__init__(nn.GroupNorm(num_groups=norm_groups, num_channels=in_chs), act_fn,
                         nn.Conv2d(in_chs, out_chs, 3, 1, 1))

# Positional embedding (for including time information)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts  = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)

class AttentionBlock(nn.Module):
    def __init__(self, channels=64, norm_groups=32, n_heads=4, dropout=0.0):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=n_heads, batch_first=True, dropout=dropout)

    def forward(self, x, *args):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
        return x + h

class NonlocalAttention(nn.Module):
    def __init__(self, channels=64, norm_groups=32, dropout=0.0, **kwargs):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.dropout_p = dropout
    
    def forward(self, x, *args):
        xnorm = self.group_norm(x)
        b, c, h, w = xnorm.shape
        # [B,C,H,W] -> [B,C,H*W]    
        q, k, v = self.q(xnorm).reshape(b,c, h*w), self.k(xnorm).reshape(b,c, h*w), self.v(xnorm).reshape(b,c, h*w)
        A = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=False) # [B, C, H*W] 
        A = A.reshape(b, c, h, w) # [B, C, H*W] -> [B, C, H, W] 
        return x + A

# Downsampling-convolutive layer
class DownSample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x, *args, **kwargs):
        return self.downsample(x)

# Upsampling-convolutive layer
class UpSample(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
    def forward(self, x, *args, **kwargs):
        return self.upsample(x)

class ResBlockWithTime(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512,
                    apply_attention=False,
                    attn_type="multihead", norm_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        attn_type = {"nonlocal":NonlocalAttention, "multihead":AttentionBlock}[attn_type]

        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=norm_groups, num_channels=self.in_channels)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=norm_groups, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()
        
        if apply_attention:
            self.attention = attn_type(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t, **kwargs):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add in timestep embedding
        h += self.dense_1(self.act_fn(t))[:, :, None, None]

        # group 3
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h

def get_from_idx(element: torch.Tensor, idx: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, idx)
    return ele.reshape(-1, 1, 1, 1)

class SimpleDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.timesteps = timesteps
        # The betas and the alphas
        self.register_buffer("beta" ,self.get_betas(beta_schedule, beta_start, beta_end))
        # Some intermediate values we will need
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("one_by_sqrt_alpha", 1. / torch.sqrt(self.alpha))
        self.register_buffer("sqrt_one_minus_alpha_bar",torch.sqrt(1 - self.alpha_bar))

    def get_betas(self, beta_schedule, beta_start, beta_end):
        if beta_schedule == "linear":
            return torch.linspace(
                beta_start,
                beta_end,
                self.timesteps,
                dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            return torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                self.timesteps,
                dtype=torch.float32) ** 2

    def forward(self, x0: torch.Tensor, timesteps: torch.Tensor):
        # Generate normal noise
        epsilon = torch.randn_like(x0)
        mean    = get_from_idx(self.sqrt_alpha_bar, timesteps) * x0      # Mean
        std_dev = get_from_idx(self.sqrt_one_minus_alpha_bar, timesteps) # Standard deviation
        # Sample is mean plus the scaled noise
        sample  = mean + std_dev * epsilon
        return sample, epsilon


