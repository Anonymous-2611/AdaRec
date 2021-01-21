import logging

import torch
from torch import nn
from torch.nn import functional as F

from models.bert.embedding.bert import BertEmbedding
from .operators import Conv1D, DilConv1D, CausalDilConv1D, Pool1D, SkipConnect, NoneConnect

ACT_FN_S = ["tanh", "relu", "gelu", "eye"]
ACT_FN_CLS = {
    "tanh": torch.tanh,
    "relu": F.relu,
    "gelu": F.gelu,
    "eye": nn.Identity(),  # force line break
}

# For nin/sas preset:
# RESIDUAL_ACT_FN = "relu"
# LOGITS_ACT_FN = "eye"

RESIDUAL_ACT_FN = "relu"
LOGITS_ACT_FN = "eye"

assert RESIDUAL_ACT_FN in ACT_FN_S
assert LOGITS_ACT_FN in ACT_FN_S


class SearchEdge(nn.Module):
    # Search one edge
    def __init__(self, num_hidden):
        super(SearchEdge, self).__init__()

        self.std_cnn_3 = Conv1D(num_hidden, num_hidden, 3)
        self.dil_cnn_3 = DilConv1D(num_hidden, num_hidden, 3, 2)
        self.cau_cnn_3 = CausalDilConv1D(num_hidden, num_hidden, 3, 2)

        self.std_cnn_5 = Conv1D(num_hidden, num_hidden, 5)
        self.dil_cnn_5 = DilConv1D(num_hidden, num_hidden, 5, 2)
        self.cau_cnn_5 = CausalDilConv1D(num_hidden, num_hidden, 5, 2)

        self.max_pool = Pool1D("max", 3)
        self.avg_pool = Pool1D("avg", 3)

        self.res = SkipConnect()
        self.none = NoneConnect()

    def forward(self, x, alpha):
        # x: [B, L, C]
        # alpha: [|O|]
        x = F.relu(x)  # [B, L, C]
        h = torch.stack(
            [
                # kernel=3
                self.std_cnn_3(x),
                self.dil_cnn_3(x),
                self.cau_cnn_3(x),
                # kernel=5
                self.std_cnn_5(x),
                self.dil_cnn_5(x),
                self.cau_cnn_5(x),
                # pool
                self.max_pool(x),
                self.avg_pool(x),
                # aux
                self.res(x),
                self.none(x),
            ]
        )  # [|O|, B, L, C]
        op_weight = torch.reshape(alpha, (-1, 1, 1, 1))  # [|O|, 1, 1, 1]
        h = torch.sum(h * op_weight, 0)  # [B, L, C]
        return h


class SearchNode(nn.Module):
    # Search all edges flow into a certain node
    def __init__(self, num_input, num_hidden):
        super(SearchNode, self).__init__()
        self.num_input = num_input

        self.edges = nn.ModuleList([SearchEdge(num_hidden) for _ in range(self.num_input)])

    def forward(self, inputs, alpha_arch_node):
        # inputs: [..., B, L, C]

        states = []
        # self.num_input == len(inputs) == len(alpha_arch_node) == len(self.edges)
        for idx in range(self.num_input):
            cur_input = inputs[idx]
            alpha = alpha_arch_node[idx]
            edge = self.edges[idx]

            cur_state = edge(cur_input, alpha)

            states.append(cur_state)
        # states: num_input * [B, L, C]
        return sum(states)


class SearchCell(nn.Module):
    # Build DAG according to alpha_arch
    def __init__(self, num_node, num_hidden):
        super(SearchCell, self).__init__()
        self.num_node = num_node

        self.nodes = nn.ModuleList([SearchNode(idx + 1, num_hidden) for idx in range(num_node)])
        self.attention_w = nn.Parameter(1e-2 * torch.randn(self.num_node), requires_grad=True)  # [num_node]
        self.ln = nn.LayerNorm(num_hidden, eps=1e-8)

        logging.info(f"Residual part activation: {RESIDUAL_ACT_FN}")

    def forward(self, x_in, alpha_arch):
        feature_maps = [x_in]
        for idx in range(self.num_node):
            node_output = self.nodes[idx](feature_maps, alpha_arch[idx])  # [B, L, C]
            feature_maps.append(node_output)
        # feature_maps : [num_node+1, B, L, C]

        attention = torch.softmax(self.attention_w, dim=0)  # [num_node]
        attention = torch.reshape(attention, (-1, 1, 1, 1))  # [num_node, 1, 1, 1]

        outputs = torch.stack(feature_maps[-self.num_node :], 0)  # [num_node, B, L, C]
        outputs = attention * outputs  # [num_node, B, L ,C]
        outputs = torch.sum(outputs, 0)  # [B, L, C]

        outputs = self.ln(outputs)

        residual_act_fn = ACT_FN_CLS[RESIDUAL_ACT_FN]
        outputs = residual_act_fn(outputs)
        # outputs = F.relu(outputs)

        return outputs + x_in


class SearchModel(nn.Module):
    def __init__(self, num_cells, num_node, vocab_size, num_teacher_hidden, num_hidden, num_class, max_len, dropout):
        super(SearchModel, self).__init__()
        self.num_cells = num_cells

        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=num_hidden, max_len=max_len, dropout=dropout)
        self.embedding_projection = nn.Linear(num_hidden, num_teacher_hidden)

        self.cells = nn.ModuleList([SearchCell(num_node, num_hidden) for _ in range(num_cells)])
        self.cells_projection = nn.Linear(num_hidden, num_teacher_hidden)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_hidden, num_class)

        logging.info(f"Logits activation: {LOGITS_ACT_FN}")

        # self._init()

    def _init(self):
        self.embedding_projection.weight.data.normal_(0.0, 0.01)
        self.embedding_projection.bias.data.fill_(0.1)
        self.cells_projection.weight.data.normal_(0.0, 0.01)
        self.cells_projection.bias.data.fill_(0.1)

    def predict(self, x, alpha_arch):
        x = self.embedding(x)
        for cell in self.cells:
            x = cell(x, alpha_arch)
        x = self.dropout(x)

        logits_act_fn = ACT_FN_CLS[LOGITS_ACT_FN]
        x = logits_act_fn(x)
        out = self.out(x)
        return out

    def forward(self, x, alpha_arch):
        # x: [B, L]

        student_emb = self.embedding(x)  # [B, L, C_s]
        aligned_student_emb = self.embedding_projection(student_emb)  # [B, L, C_t]

        aligned_cells_out = []
        x = student_emb
        for cell in self.cells:
            x = cell(x, alpha_arch)  # [B, L, C_s]
            aligned_x = self.cells_projection(x)  # [B, L, C_t]
            aligned_cells_out.append(aligned_x)
        # aligned_cells_out : (num_cells) * [B, L, C_t]

        x = self.dropout(x)

        logits_act_fn = ACT_FN_CLS[LOGITS_ACT_FN]
        x = logits_act_fn(x)
        out = self.out(x)  # [B, L, num_class]

        return aligned_student_emb, aligned_cells_out, out
