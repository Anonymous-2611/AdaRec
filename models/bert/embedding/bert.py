import torch.nn as nn

from .position import PositionalEmbedding
from .token import TokenEmbedding


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def load_embs(self, emb_state_dict):
        self.load_state_dict(emb_state_dict)

    def forward(self, input_seqs):
        # input_seqs: [B, L]
        x = self.token(input_seqs) + self.position(input_seqs)  # [B, L, C]
        return self.dropout(x)
