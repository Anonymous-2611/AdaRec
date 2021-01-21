import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.bert.embedding.bert import BertEmbedding
from .operators import OPERATOR_CLS, OPERATOR_NAME


class FinetuneEdge(nn.Module):
    def __init__(self, num_hidden, alpha):
        super(FinetuneEdge, self).__init__()

        op_index = np.argmax(alpha)

        self.op_name = OPERATOR_NAME[op_index]
        self.add_module(self.op_name, OPERATOR_CLS[op_index](num_hidden))

    def forward(self, x):
        x = F.relu(x)  # [B, L, C]
        h = self.__getattr__(self.op_name)(x)  # [B, L, C]
        return h


class FinetuneNode(nn.Module):
    def __init__(self, num_hidden, alphas):
        super(FinetuneNode, self).__init__()

        self.edges = nn.ModuleList([FinetuneEdge(num_hidden, alpha) for alpha in alphas])

    def forward(self, inputs):
        # inputs: [..., B, L, C]
        # len(inputs) == len(self.edges)
        states = []
        for edge, edge_input in zip(self.edges, inputs):
            cur_state = edge(edge_input)
            states.append(cur_state)
        return sum(states)  # [B, L, C]


class FinetuneCell(nn.Module):
    def __init__(self, num_hidden, num_node, alpha_arch):
        super(FinetuneCell, self).__init__()
        self.num_node = num_node

        self.nodes = nn.ModuleList([FinetuneNode(num_hidden, alpha_arch[idx]) for idx in range(self.num_node)])
        self.attention_w = nn.Parameter(1e-2 * torch.randn(self.num_node), requires_grad=True)  # [num_node]
        self.ln = nn.LayerNorm(num_hidden, eps=1e-8)

    def forward(self, x_in):
        feature_maps = [x_in]
        for node in self.nodes:
            node_output = node(feature_maps)  # [B, L, C]
            feature_maps.append(node_output)
        # feature_maps: [num_node+1, B, L, C]

        attention = torch.softmax(self.attention_w, dim=0)  # [num_node]
        attention = torch.reshape(attention, (-1, 1, 1, 1))  # [num_node, 1, 1, 1]

        outputs = torch.stack(feature_maps[-self.num_node :], 0)  # [num_node, B, L, C]
        outputs = attention * outputs  # [num_node, B, L ,C]
        outputs = torch.sum(outputs, 0)  # [B, L, C]

        outputs = self.ln(outputs)
        outputs = F.relu(outputs)

        return outputs + x_in


class FinetuneModel(nn.Module):
    def __init__(self, alpha_arch, num_cells, num_node, vocab_size, num_hidden, num_class, max_len, dropout):
        super(FinetuneModel, self).__init__()

        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=num_hidden, max_len=max_len, dropout=dropout)
        self.cells = nn.ModuleList([FinetuneCell(num_hidden, num_node, alpha_arch) for _ in range(num_cells)])

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_hidden, num_class)

    def forward(self, x):
        x = self.embedding(x)  # [B, L, C]

        for cell in self.cells:
            x = cell(x)  # [B, L, C]

        x = self.dropout(x)
        out = self.out(x)  # [B, L, num_class]
        return out
