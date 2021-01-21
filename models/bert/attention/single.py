import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # query : [B, H, L, C']
        # key   : [B, H, L, C']
        # value : [B, H, L, C']
        # mask  : [B, 1, L, L]

        # [B, H, L, C'] * [B, H, C', L] => [B, H, L, L]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # [B, H, L, L]

        p_attn = F.softmax(scores, dim=-1)  # [B, H, L, L]

        if dropout is not None:
            p_attn = dropout(p_attn)

        # [B, H, L, L] * [B, H, L, C'] => [B, H, L, C']
        return torch.matmul(p_attn, value), p_attn
