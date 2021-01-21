from torch import nn

from .modules.finetune import FinetuneModel


class StudentFinetune(nn.Module):
    def __init__(self, args, alpha_arch):
        super(StudentFinetune, self).__init__()

        self.max_len = args.loader_max_len
        self.num_class = args.loader_num_items + 1
        self.vocab_size = args.loader_num_items + args.loader_num_aux_vocabs

        self.dropout = args.model_dropout
        self.num_hidden = args.model_num_hidden  # Embedding channel
        self.num_node = args.model_num_node  # Number of intermediate nodes in a cell
        self.num_cell = args.model_num_cell

        self.alpha_arch = alpha_arch

        self.net = FinetuneModel(
            alpha_arch,
            num_cells=self.num_cell,
            num_node=self.num_node,
            vocab_size=self.vocab_size,
            num_hidden=self.num_hidden,
            num_class=self.num_class,
            max_len=self.max_len,
            dropout=self.dropout,
        )

    def net_parameters(self):
        return self.net.parameters()

    def forward(self, input_seqs):
        # input
        #   - input_seqs : [B, L]

        out = self.net(input_seqs)  # [B, L, num_class]
        return out
