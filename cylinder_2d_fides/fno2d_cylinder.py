"""
fno2d_cylinder.py - FNO for cylinder flow (3 channels: omega, ux, uy).

Same architecture as the cavity FNO but with 3 input/output channels
and grid dimensions (150, 50) instead of (64, 64).
"""

import torch
import torch.nn as nn
import numpy as np


class SpectralConv2d(nn.Module):
    """2D Fourier spectral convolution layer."""

    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))

    def _complex_mul(self, a, b):
        """Complex multiplication using real/imag pairs."""
        real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        B = x.shape[0]
        # FFT
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        mx, my = self.modes_x, self.modes_y
        out_ft = torch.zeros(B, self.out_channels, x.shape[2], x.shape[3] // 2 + 1, 2,
                              device=x.device, dtype=x.dtype)

        # Low frequency modes
        out_ft[:, :, :mx, :my] = torch.einsum('bixyz,ioxyz->boxyz',
                                                x_ft[:, :, :mx, :my], self.weights1)
        out_ft[:, :, -mx:, :my] = torch.einsum('bixyz,ioxyz->boxyz',
                                                 x_ft[:, :, -mx:, :my], self.weights2)

        # Back to physical space
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        return torch.fft.irfft2(out_ft_complex, s=(x.shape[2], x.shape[3]))


class FNOBlock(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super().__init__()
        self.sp = SpectralConv2d(width, width, modes_x, modes_y)
        self.sk = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return torch.relu(self.sp(x) + self.sk(x))


class CylinderFNO(nn.Module):
    """
    FNO for cylinder flow ROM.

    Input:  (B, ic, nx, ny) - coarse state [omega, ux, uy]
    Output: (B, oc, nx, ny) - fine state [omega, ux, uy]
    """

    def __init__(self, ic=3, oc=3, w=32, nb=4, mx=12, my=8):
        super().__init__()
        self.lift = nn.Conv2d(ic, w, 1)
        self.blocks = nn.ModuleList([FNOBlock(w, mx, my) for _ in range(nb)])
        self.p1 = nn.Conv2d(w, w * 2, 1)
        self.p2 = nn.Conv2d(w * 2, oc, 1)

    def forward(self, x):
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        x = torch.relu(self.p1(x))
        return self.p2(x)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CylinderFNO(ic=3, oc=3, w=32, nb=4, mx=12, my=8).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f'CylinderFNO: {nparams:,} parameters')

    x = torch.randn(2, 3, 150, 50, device=device)
    y = model(x)
    print(f'Input: {x.shape} -> Output: {y.shape}')
