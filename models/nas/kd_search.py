import logging
import numpy as np

from pyemd import emd_with_flow
from utils.others import np_softmax

import torch
from torch import nn
from torch.nn import functional as F

from .modules.funcs import get_logits
from .modules.layer_select import layer_select
from .modules.operators import OPERATOR_NAME, OPERATOR_COST
from .modules.search import SearchModel


class StudentSearch(nn.Module):
    """
    Notation:
      B   : batch size
      L   : length of sequence
      C   : hidden channels (number of hidden)
      |O| : number of operators (types of edge)
      |S| : number of NAS-cells (number of layers in student network)
      |T| : number of layers in teacher network
    """

    def __init__(self, args):
        super(StudentSearch, self).__init__()
        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)  # get device later on

        self.num_cells = args.model_num_cell

        self.teacher_layers = args.search_teacher_layers
        self.teacher_num_hidden = args.search_teacher_hidden

        self.student_layers = args.model_num_cell
        self.distill_loss_type = args.search_distill_loss.lower()

        self.temperature_init = args.search_temperature
        self.temperature_decay_rate = args.search_temperature_decay_rate  # Decay temperature by rate
        self.temperature_decay_epochs = args.search_temperature_decay_epochs  # Decay temperature every n epochs

        self.max_len = args.loader_max_len
        self.num_class = args.loader_num_items + 1
        self.vocab_size = args.loader_num_items + args.loader_num_aux_vocabs

        self.dropout = args.model_dropout
        self.num_node = args.model_num_node  # Number of intermediate nodes in a cell
        self.num_hidden = args.model_num_hidden  # Embedding channel

        self.num_operator = len(OPERATOR_NAME)

        # [Search space]
        self.alpha_arch = nn.ParameterList()
        # for layer in range(2, 2 + self.num_node):
        for layer in range(1, 1 + self.num_node):
            self.alpha_arch.append(nn.Parameter(1e-2 * torch.randn(layer, self.num_operator), requires_grad=True))

        self.emd_temperature = 10
        self.teacher_weights = np.ones(self.teacher_layers) / self.teacher_layers
        self.student_weights = np.ones(self.student_layers) / self.student_layers
        self.mse = nn.MSELoss()

        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchModel(
            num_cells=self.num_cells,
            num_node=self.num_node,
            vocab_size=self.vocab_size,
            num_hidden=self.num_hidden,
            num_teacher_hidden=self.teacher_num_hidden,
            num_class=self.num_class,
            max_len=self.max_len,
            dropout=self.dropout,
        )

        self.ce = nn.CrossEntropyLoss(ignore_index=0)  # ignore zero index [PAD]

        self.global_epochs = 0
        self.temperature = self.temperature_init  # Initial temperature

        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.kl_temperature = 1.0

    def alpha_parameters(self):
        for n, p in self._alphas:
            yield p

    def net_parameters(self):
        return self.net.parameters()

    def log_layer_mapping(self):
        logging.info(f"Hierarchical distillation method: {self.layer_select_method}")
        logging.info("Layer mappings:")
        s = "\t"
        for a, b in layer_select(self.student_layers, self.teacher_layers, self.layer_select_method):
            s += "\t{:2d}<-{:2d}".format(a, b)
        logging.info(s)

    def temperature_step(self):
        # Let temperature do exponential decay, from 1 to ~0
        self.global_epochs += 1

        multiplier = self.temperature_decay_rate ** int(self.global_epochs / self.temperature_decay_epochs)
        self.temperature = self.temperature_init * multiplier

    def get_mixed_alpha_arch(self):
        # self.alpha_arch:
        #   - [1, num_ops]
        #   - [2, num_ops]
        #   ...

        mixed_alpha_arch = []
        for arch_of_layer in self.alpha_arch:
            logits = get_logits(arch_of_layer)
            mixed = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            # mixed = F.gumbel_softmax(F.log_softmax(arch_of_layer, dim=1), tau=self.temperature, hard=True, dim=-1)
            mixed_alpha_arch.append(mixed)
        return mixed_alpha_arch

    def compute_loss_cross_entropy(self, out_logits, label_seqs):
        # out_logits : [B, L, num_classes]
        # label_seqs : [B, L]

        label_flatten = label_seqs.view(-1)  # [B*L]
        logits = out_logits.view(-1, out_logits.size(-1))  # [B*L, num_classes]
        ce_loss = self.ce(logits, label_flatten)

        return ce_loss

    def layer_distill_loss(self, teacher, student):
        # [KL divergence]
        return self.kl_div(
            F.log_softmax(student / self.kl_temperature, dim=-1),  # log_p[student]
            F.softmax(teacher / self.kl_temperature, dim=-1),  # p[teacher]
        )

    def emd_distill_loss(self, layer_wise_hidden, teacher_layer_wise_hidden):
        # 1. Compute EMD loss
        # 2. Update weight

        student_weight = self.student_weights.copy()  # [|S|]
        teacher_weight = self.teacher_weights.copy()  # [|T|]

        student_weight_hist = np.concatenate((student_weight, np.zeros(self.teacher_layers)))  # [|S|+|T|]
        teacher_weight_hist = np.concatenate((np.zeros(self.student_layers), teacher_weight))  # [|S|+|T|]

        total = self.teacher_layers + self.student_layers
        distance_matrix = torch.zeros([total, total], device=self.dummy.device)

        # Compute distance matrix, shape=(|S|, |T|)
        for i in range(self.student_layers):
            student_hidden = layer_wise_hidden[i]  # [B, L, C]
            for j in range(self.teacher_layers):
                teacher_hidden = teacher_layer_wise_hidden[j]  # [B, L, C]
                # KL Div
                distance_matrix[i][j + self.student_layers] = self.layer_distill_loss(student_hidden, teacher_hidden)
                distance_matrix[j + self.student_layers][i] = self.layer_distill_loss(teacher_hidden, student_hidden)
                # MSE symmetric
                # distance = self.mse(student_hidden, teacher_hidden)
                # distance_matrix[i][j + self.student_layers] = distance
                # distance_matrix[j + self.student_layers][i] = distance

        d_np = distance_matrix.detach().cpu().numpy().astype("float64")  # [|S|+|T|, |S|+|T|]
        _, transfer_matrix = emd_with_flow(student_weight_hist, teacher_weight_hist, d_np)
        transfer_matrix = np.array(transfer_matrix, dtype=np.float)  # [|S|+|T|, |S|+|T|]

        transfer_matrix_torch = torch.tensor(transfer_matrix, device=self.dummy.device)
        kd_loss = torch.sum(transfer_matrix_torch * distance_matrix)

        # Update weight
        def update_weight(weight, t_mat, num_layers, bias=0):
            # t_mat: [|S|+|T|, |S|+|T|]
            transfer_weight = np.sum(t_mat * d_np, -1)  # [|S|+|T|]
            for idx in range(num_layers):
                weight[idx] = transfer_weight[idx + bias] / weight[idx]
            weight_sum = np.sum(weight)
            for idx in range(num_layers):
                if weight[idx] != 0:
                    weight[idx] = weight_sum / weight[idx]
            weight = np_softmax(weight / self.emd_temperature)
            return weight

        self.student_weights = update_weight(student_weight, transfer_matrix, self.student_layers)
        self.teacher_weights = update_weight(
            teacher_weight, np.transpose(transfer_matrix), self.teacher_layers, bias=self.student_layers
        )

        return kd_loss

    def compute_loss_distill(self, emb_s, emb_t, hidden_s, hidden_t, out_s, out_t):
        loss_embedding = self.mse(emb_s, emb_t)
        loss_hidden = self.emd_distill_loss(hidden_s, hidden_t)
        loss_logits = self.layer_distill_loss(out_s, out_t)
        return loss_embedding, loss_hidden, loss_logits

    def compute_loss_model_efficiency(self, alpha_arch_mixed):
        edge_efficiency_losses = []

        cost_weight = torch.from_numpy(OPERATOR_COST).to(self.dummy.device)  # [|O|]
        for alphas in alpha_arch_mixed:
            for alpha in alphas:  # [|O|]
                cost = alpha * cost_weight
                edge_efficiency_losses.append(torch.sum(cost, 0))
        # edge_efficiency_losses: (N+1)*N/2 * scalar

        model_efficiency_loss = sum(edge_efficiency_losses)

        return model_efficiency_loss

    def predict(self, input_seqs):
        alpha_arch_mixed = self.get_mixed_alpha_arch()

        # out : [B, L, num_class]
        out = self.net.predict(input_seqs, alpha_arch_mixed)
        return out

    def forward(self, input_seqs, label_seqs, teacher_emb, teacher_hidden, teacher_out):
        # input
        #   - input_seqs     : [B, L]
        #   - label_seqs     : [B, L]
        #   - teacher_emb    : [B, L, C_t]
        #   - teacher_hidden : [|T|, B, L, C_t]
        #   - teacher_out    : [B, L, num_class]

        # intermediate
        #   - alpha_arch_mixed    : (num_node) * [..., |O|]
        #   - aligned_student_emb : [B, L, C_t] (using linear projection)
        #   - aligned_cells_out   : (|S|) * [B, L, C_t] (using linear projection)
        #   - out                 : [B, L, num_class]

        # output
        #   - loss_task
        #   - loss_distill : (loss_embedding, loss_hidden, loss_logits)
        #   - loss_complexity
        alpha_arch_mixed = self.get_mixed_alpha_arch()

        aligned_student_emb, aligned_cells_out, student_out = self.net(input_seqs, alpha_arch_mixed)

        # 1. task
        loss_task = self.compute_loss_cross_entropy(student_out, label_seqs)

        # 2. distillation
        pack = (aligned_student_emb, teacher_emb, aligned_cells_out, teacher_hidden, student_out, teacher_out)
        loss_distill = self.compute_loss_distill(*pack)

        # 3. efficiency aware
        loss_complexity = self.compute_loss_model_efficiency(alpha_arch_mixed)

        return loss_task, loss_distill, loss_complexity
