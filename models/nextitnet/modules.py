from torch import nn
import torch

from torch.nn import functional as F


class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, kernel_size=1, causal=True):
        super(Conv1d, self).__init__()
        self.causal = causal

        if self.causal:
            self.padding = [0, 0, (kernel_size - 1) * dilation, 0, 0, 0]
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, dilation=dilation)
        else:
            padding = (kernel_size - 1) * dilation // 2
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, dilation=dilation, padding=padding)

    def forward(self, x):
        # x: [B, L, C]
        if self.causal:
            x = F.pad(x, self.padding)  # [B, (K-1)*D+L, C]

        # | causal     : [B, (K-1)*D+L, C]
        # | non-causal : [B,         L, C]
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # => [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]

        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_hidden, kernel_size, dilation, use_eps=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv1d(num_hidden, num_hidden, dilation, kernel_size, causal=True)
        self.ln1 = nn.LayerNorm(num_hidden, eps=1e-8)

        self.conv2 = Conv1d(num_hidden, num_hidden, 2 * dilation, kernel_size, causal=True)
        self.ln2 = nn.LayerNorm(num_hidden, eps=1e-8)

        if use_eps:
            self.eps = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.eps = 1.0

    def forward(self, x_in):
        # x_in: [B, L, C]
        x = self.conv1(x_in)
        x = F.relu(self.ln1(x))
        x = self.conv2(x)
        x = F.relu(self.ln2(x))
        return x_in + self.eps * x  # [B, L, C]
