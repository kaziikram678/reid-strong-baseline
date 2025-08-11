# layers/self_attention.py
import torch
import torch.nn as nn

class SelfAttention2d(nn.Module):
    """
    Simple non-local style self-attention for 2D feature maps.
    Input:  (B, C, H, W)
    Output: (B, C, H, W) with residual
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        inter = max(1, in_channels // reduction)

        self.query = nn.Conv2d(in_channels, inter, kernel_size=1, bias=False)
        self.key   = nn.Conv2d(in_channels, inter, kernel_size=1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma   = nn.Parameter(torch.zeros(1))  # learnable residual weight

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)   # (B, N, inter)
        k = self.key(x).view(B, -1, N)                      # (B, inter, N)
        attn = torch.bmm(q, k)                              # (B, N, N)
        attn = self.softmax(attn)

        v = self.value(x).view(B, C, N)                     # (B, C, N)
        out = torch.bmm(v, attn.permute(0, 2, 1))           # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x                         # residual
