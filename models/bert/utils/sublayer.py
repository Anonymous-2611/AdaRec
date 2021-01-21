import torch.nn as nn
import torch


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, use_eps):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        if use_eps:
            self.eps = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.eps = 1.0

    def forward(self, x, sublayer_fn):
        # x: [B, L, C]
        return x + self.eps * self.dropout(sublayer_fn(self.norm(x)))
