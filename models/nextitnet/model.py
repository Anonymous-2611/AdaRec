from torch import nn
from math import sqrt

from models.nextitnet.modules import ResidualBlock


class NextItNet(nn.Module):
    def __init__(self, args, is_student=False, num_teacher_hidden=None):
        super(NextItNet, self).__init__()
        # Should I use masked language model in NextItNet or SASRec?

        self.is_student = is_student

        self.num_items = args.loader_num_items + 1  # + [PAD=0]
        self.vocab_size = args.loader_num_items + args.loader_num_aux_vocabs

        self.group_size = len(args.nin_block_dilations)
        self.dilations = args.nin_block_dilations * args.nin_num_blocks
        self.num_hidden = args.nin_hidden_units
        self.kernel_size = args.nin_kernel_size
        self.use_eps = args.nin_use_eps

        self.embedding = nn.Embedding(self.vocab_size, self.num_hidden)
        self.blocks = nn.ModuleList([self.make_block(dilation) for dilation in self.dilations])
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

    def _init(self):
        stdv = sqrt(1.0 / self.num_items)
        self.embedding.weight.data.uniform_(-stdv, stdv)

        self.out.weight.data.normal_(0.0, 0.01)
        self.out.bias.data.fill_(0.1)

    def make_block(self, dilation):
        return ResidualBlock(
            num_hidden=self.num_hidden, kernel_size=self.kernel_size, dilation=dilation, use_eps=self.use_eps
        )

    # def logits_and_hidden(self, input_seqs):
    #     # input_seqs: [B, L]
    #     x = self.embedding(input_seqs)  # [B, L, C]
    #
    #     hidden = []
    #     for block in self.blocks:
    #         x = block(x)  # [B, L, C]
    #         hidden.append(x.detach().clone())
    #     # hidden_list: (num_blocks) * [B, L, C]
    #
    #     out = self.out(x)  # [B, L, num_classes]
    #     return out, hidden

    def emb_hidden_logits(self, input_seqs):
        # input_seqs: [B, L]
        emb = self.embedding(input_seqs)  # [B, L, C]

        hidden = []
        x = emb
        for idx, block in enumerate(self.blocks):
            x = block(x)  # [B, L, C]
            if (idx + 1) % self.group_size == 0:
                # [1,4] -> [1,4] -> ..., just record last blocks' output
                if self.is_student:
                    hidden.append(self.hid_proj(x))  # [B, L, C_t]
                else:
                    hidden.append(x)

        logits = self.out(x)  # [B, L, num_classes]
        if self.is_student:
            emb = self.emb_proj(emb)  # [B, L, C_t]
        return emb, hidden, logits

    def forward(self, input_seqs):
        # input_seqs: [B, L]
        x = self.embedding(input_seqs)  # [B, L, C]

        for block in self.blocks:
            x = block(x)  # [B, L, C]

        out = self.out(x)  # [B, L, num_classes]
        return out
