from torch import nn

from .embedding.bert import BertEmbedding
from .transformer import TransformerBlock


class BertModel(nn.Module):
    def __init__(self, args, is_student=False, num_teacher_hidden=None):
        super().__init__()
        self.bert = BertEncoder(args, is_student, num_teacher_hidden)
        # +1 => [PAD]
        self.out = nn.Linear(args.bert_hidden_units, args.loader_num_items + 1)

    def emb_hidden_logits(self, input_seqs):
        emb, hidden, last = self.bert.emb_hidden_last(input_seqs)
        logits = self.out(last)
        return emb, hidden, logits

    def forward(self, input_seqs):
        # input_seqs: [B, C]
        encode = self.bert(input_seqs)
        return self.out(encode)  # [B, L, num_classes]


class BertEncoder(nn.Module):
    def __init__(self, args, is_student=False, num_teacher_hidden=None):
        super().__init__()

        self.is_student = is_student

        self.max_len = args.loader_max_len
        self.vocab_size = args.loader_num_items + args.loader_num_aux_vocabs

        self.dropout = args.bert_dropout
        self.num_hidden = args.bert_hidden_units
        self.num_heads = args.bert_num_heads
        self.num_blocks = args.bert_num_blocks
        self.use_eps = args.bert_use_eps

        self.embedding = BertEmbedding(
            vocab_size=self.vocab_size, embed_size=self.num_hidden, max_len=self.max_len, dropout=self.dropout
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.num_hidden, self.num_heads, self.num_hidden * 4, self.dropout, self.use_eps)
                for _ in range(self.num_blocks)
            ]
        )
        if self.is_student:
            assert num_teacher_hidden is not None
            if num_teacher_hidden == self.num_hidden:
                self.emb_proj = nn.Identity()
                self.hid_proj = nn.Identity()
            else:
                self.emb_proj = nn.Linear(self.num_hidden, num_teacher_hidden)
                self.hid_proj = nn.Linear(self.num_hidden, num_teacher_hidden)

    def emb_hidden_last(self, x):
        # x: [B, L]
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # [B, 1, L, L]
        emb = self.embedding(x)  # [B, L, C]

        hidden = []
        h = emb
        for block in self.blocks:
            h = block(h, mask)  # [B, L, C]
            if self.is_student:
                hidden.append(self.hid_proj(h))  # [B, L, C_t]
            else:
                hidden.append(h)
        last = h
        if self.is_student:
            emb = self.emb_proj(emb)  # [B, L, C_t]
        return emb, hidden, last

    def forward(self, x):
        # x: [B, L]
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # [B, 1, L, L]
        x = self.embedding(x)  # [B, L, C]

        for block in self.blocks:
            x = block.forward(x, mask)  # [B, L, C]
        return x
