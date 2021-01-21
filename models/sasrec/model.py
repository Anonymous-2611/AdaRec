import torch
from torch import nn
from math import sqrt

from models.sasrec.modules import SASBlock


class SASRec(nn.Module):
    def __init__(self, args, is_student=False, num_teacher_hidden=None):
        super(SASRec, self).__init__()

        self.is_student = is_student

        self.max_len = args.loader_max_len
        self.num_items = args.loader_num_items + 1  # + [PAD=0]
        self.vocab_size = args.loader_num_items + args.loader_num_aux_vocabs

        self.num_blocks = args.sas_num_blocks
        self.num_hidden = args.sas_hidden_units
        self.num_heads = args.sas_num_heads

        self.dropout = args.sas_dropout
        self.use_eps = args.sas_use_eps

        self.token_embedding = nn.Embedding(self.vocab_size, self.num_hidden)
        self.pos_embedding = nn.Embedding(self.max_len, self.num_hidden)

        self.blocks = nn.ModuleList([self.make_block() for _ in range(self.num_blocks)])
        self.ln = nn.LayerNorm(self.num_hidden, eps=1e-8)
        self.out = nn.Linear(self.num_hidden, self.num_items)

        if self.is_student:
            assert num_teacher_hidden is not None
            if num_teacher_hidden == self.num_hidden:
                self.emb_proj = nn.Identity()
                self.hid_proj = nn.Identity()
            else:
                self.emb_proj = nn.Linear(self.num_hidden, num_teacher_hidden)
                self.hid_proj = nn.Linear(self.num_hidden, num_teacher_hidden)

        self._init()
        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)  # get device later on

    def make_block(self):
        return SASBlock(self.num_hidden, self.num_heads, self.dropout, self.use_eps)

    def _init(self):
        stdv = sqrt(1.0 / self.num_items)
        self.token_embedding.weight.data.uniform_(-stdv, stdv)

        stdv = sqrt(1.0 / self.max_len)
        self.pos_embedding.weight.data.uniform_(-stdv, stdv)

        self.out.weight.data.normal_(0.0, 0.01)
        self.out.bias.data.fill_(0.1)

    # def logits_and_hidden(self, x_in):
    #     # x_in: [B, L]
    #     pos_indices = torch.arange(self.max_len).to(self.dummy.device)
    #
    #     token_emb = self.token_embedding(x_in)  # [B, L, C]
    #     pos_emb = self.pos_embedding(pos_indices)  # [B, L, C]
    #     x = token_emb + pos_emb  # [B, L, C]
    #
    #     hidden = []
    #     for block in self.blocks:
    #         x = block(x)  # [B, L, C]
    #         hidden.append(x.detach().clone())
    #     x = self.ln(x)
    #     out = self.out(x)
    #
    #     return out, hidden

    def emb_hidden_logits(self, input_seqs):
        pos_indices = torch.arange(self.max_len).to(self.dummy.device)

        token_emb = self.token_embedding(input_seqs)  # [B, L, C]
        pos_emb = self.pos_embedding(pos_indices)  # [B, L, C]

        emb = token_emb + pos_emb  # [B, L, C]
        hidden = []
        x = emb
        for block in self.blocks:
            x = block(x)  # [B, L, C]
            if self.is_student:
                hidden.append(self.hid_proj(x))
            else:
                hidden.append(x)

        x = self.ln(x)
        logits = self.out(x)
        if self.is_student:
            emb = self.emb_proj(emb)

        return emb, hidden, logits

    def forward(self, input_seqs):
        # x_in: [B, L]
        pos_indices = torch.arange(self.max_len).to(self.dummy.device)

        token_emb = self.token_embedding(input_seqs)  # [B, L, C]
        pos_emb = self.pos_embedding(pos_indices)  # [B, L, C]

        x = token_emb + pos_emb  # [B, L, C]
        for block in self.blocks:
            x = block(x)  # [B, L, C]

        x = self.ln(x)
        out = self.out(x)

        return out
