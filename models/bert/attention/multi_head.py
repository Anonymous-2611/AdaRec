import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        assert d_model % h == 0
        self.d_k = d_model // h

        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query : [B, L, C]
        # key   : [B, L, C]
        # value : [B, L, C]
        # mask  : [B, 1, L, L]

        batch_size = query.size(0)

        query, key, value = [
            # C' = C // H
            # [B, L, C] => [B, L, C] => [B, L, H, C'] => [B, H, L, C']
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        # query : [B, H, L, C']
        # key   : [B, H, L, C']
        # value : [B, H, L, C']

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # x     : [B, H, L, C']
        # attn  : [B, H, L, L]

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # [B, H, L, C'] => [B, L, H, C'] => [B, L, C]

        return self.output_linear(x)  # [B, L, C]
