import torch.nn as nn

from .attention.multi_head import MultiHeadedAttention
from .utils.feed_forward import PositionWiseFeedForward
from .utils.sublayer import SublayerConnection


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, use_eps):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout, use_eps=use_eps)

        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout, use_eps=use_eps)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x:    [B, L, C]
        # mask: [B, 1, L, L]

        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))  # [B, L, C]
        x = self.output_sublayer(x, self.feed_forward)  # [B, L, C]
        return self.dropout(x)  # [B, L, C]
