# layers/sgconv.py
import torch
import torch.nn as nn

def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)

class SGConv2D(nn.Module):
    """
    SGConv for CNN feature maps.
    Input : (B, C_in, H, W)
    Output: (B, C_out, H, W)
    A_base: 4-neighbor grid adjacency (with self loops), built once per (H, W).
    """
    def __init__(self, dim_in, dim_out, thresh=0.01):
        super().__init__()
        self.phi = nn.Linear(dim_in, dim_in, bias=False)
        self.psi = nn.Linear(dim_in, dim_in, bias=False)
        self.W0  = nn.Linear(dim_in, dim_out, bias=False)
        self.W1  = nn.Linear(dim_in, dim_out, bias=False)
        self.act = nn.ReLU()
        for l in (self.phi, self.psi, self.W0, self.W1):
            weights_init_kaiming(l)

        self.thresh = thresh
        self._hw = None
        self.register_buffer("A_base", None)  # (N, N)
        self.register_buffer("I", None)       # (N, N)

    @torch.no_grad()
    def _build_grid_adj(self, H, W, device):
        N = H * W
        A = torch.eye(N, device=device)  # include self-loops
        def idx(h, w): return h * W + w
        for h in range(H):
            for w in range(W):
                i = idx(h, w)
                if h > 0:     A[i, idx(h - 1, w)] = 1
                if h < H - 1: A[i, idx(h + 1, w)] = 1
                if w > 0:     A[i, idx(h, w - 1)] = 1
                if w < W - 1: A[i, idx(h, w + 1)] = 1
        self.A_base = A   # (N, N)
        self.I = torch.eye(N, device=device)
        self._hw = (H, W)

    def forward(self, x):
        # x: (B, C_in, H, W)
        B, C, H, W = x.shape
        if self._hw != (H, W) or self.A_base is None:
            self._build_grid_adj(H, W, x.device)

        N = H * W
        X = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # (B, N, C)

        # S = softmax(phi(X) @ psi(X)^T) with tiny thresholding
        Q = self.phi(X)                       # (B, N, C)
        K = self.psi(X)                       # (B, N, C)
        S = torch.bmm(Q, K.transpose(1, 2))   # (B, N, N)
        S = torch.softmax(S, dim=2)
        if self.thresh > 0:
            S = S.masked_fill(S < self.thresh, 0.0)

        # A = base (grid) + S
        A = self.A_base.unsqueeze(0).expand(B, -1, -1) + S    # (B, N, N)

        # self part: diag(A)
        diagA = torch.diagonal(A, dim1=1, dim2=2)             # (B, N)
        m0 = self.W0(X * diagA.unsqueeze(-1))                 # (B, N, C_out)

        # neighbor part: (A with zero diag) @ X
        A_off = A * (1.0 - self.I.unsqueeze(0))               # (B, N, N)
        nbr_msg = torch.bmm(A_off, X)                         # (B, N, C)
        m1 = self.W1(nbr_msg)                                 # (B, N, C_out)

        Y = self.act(m0 + m1).view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return Y
