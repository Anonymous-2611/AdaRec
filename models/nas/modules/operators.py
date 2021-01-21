import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.model_stat import get_model_complexity_info

"""
0:  std_cnn_3              0.56531834
1:  dil_cnn_3              0.56531834
2:  cau_cnn_3              0.56531834
3:  std_cnn_5              2.12701275
4:  dil_cnn_5              2.12701275
5:  cau_cnn_5              2.12701275
6:  max_pool (SAME)        0.48075444
7:  avg_pool (SAME)        0.48075444
8:  skip_connect           0.48074893
9:  none                   0.48074893
"""

OPERATOR_NAME = [
    # kernel=3
    "std_cnn_3",
    "dil_cnn_3",
    "cau_cnn_3",
    # kernel=5
    "std_cnn_5",
    "dil_cnn_5",
    "cau_cnn_5",
    # pool 3
    "max_pool",
    "avg_pool",
    # aux
    "skip_connect",
    "none",
]

OPERATOR_COST = np.array(
    [
        0.56531834,
        0.56531834,
        0.56531834,
        2.12701275,
        2.12701275,
        2.12701275,
        0.48075444,
        0.48075444,
        0.48074893,
        0.48074893,
    ],
    dtype=np.float,
)

OPERATOR_CLS = [
    # kernel=3
    lambda C: Conv1D(C, C, 3),
    lambda C: DilConv1D(C, C, 3, 2),
    lambda C: CausalDilConv1D(C, C, 3, 2),
    # kernel=5
    lambda C: Conv1D(C, C, 5),
    lambda C: DilConv1D(C, C, 5, 2),
    lambda C: CausalDilConv1D(C, C, 5, 2),
    # pool
    lambda C: Pool1D("max", 3),
    lambda C: Pool1D("avg", 3),
    # aux
    lambda C: SkipConnect(),
    lambda C: NoneConnect(),
]


class Conv1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Conv1D, self).__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x


class DilConv1D(nn.Module):
    # non-causal dilated conv
    def __init__(self, in_channel, out_channel, kernel_size, dilation):
        super(DilConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channel, out_channel, kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x


class CausalDilConv1D(nn.Module):
    # casual dilated conv
    def __init__(self, in_channel, out_channel, kernel_size, dilation):
        super(CausalDilConv1D, self).__init__()

        self.padding = [0, 0, (kernel_size - 1) * dilation, 0, 0, 0]
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        # x: [B, L, C]
        x = F.pad(x, self.padding)  # [B, (K-1)*D+L, C]
        x = x.permute(0, 2, 1)  # [B, C, (K-1)*D+L]
        x = self.conv(x)  # [B, C, L]
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # [B, L, C]
        return x


class Pool1D(nn.Module):
    def __init__(self, mode, kernel_size):
        super(Pool1D, self).__init__()
        if mode.lower() == "max":
            self.pool = nn.MaxPool1d(kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        elif mode.lower() == "avg":
            self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x


class SkipConnect(nn.Identity):
    def __init__(self):
        super(SkipConnect, self).__init__()


class NoneConnect(nn.Module):
    def __init__(self):
        super(NoneConnect, self).__init__()

    def forward(self, x):
        return torch.zeros_like(x)


if __name__ == "__main__":

    def softmax(x):
        x = np.array(x)
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    seq_length = 10
    num_hidden = 32
    input_shape = (seq_length, num_hidden)

    ops = [
        # kernel=3
        ("std_cnn_3", Conv1D(num_hidden, num_hidden, 3)),
        ("dil_cnn_3", DilConv1D(num_hidden, num_hidden, 3, 2)),
        ("cau_cnn_3", CausalDilConv1D(num_hidden, num_hidden, 3, 2)),
        # kernel=3
        ("std_cnn_5", Conv1D(num_hidden, num_hidden, 5)),
        ("dil_cnn_5", DilConv1D(num_hidden, num_hidden, 5, 2)),
        ("cau_cnn_5", CausalDilConv1D(num_hidden, num_hidden, 5, 2)),
        # pool
        ("max_pool", Pool1D("max", 3)),
        ("avg_pool", Pool1D("avg", 3)),
        # aux
        ("res", SkipConnect()),
        ("none", NoneConnect()),
    ]

    # mac_plus_params = []
    MAC_l = []
    PARAM_l = []
    for name, net in ops:
        print("-+-" * 10)
        print(f"Operator: {name}")
        MACs, params = get_model_complexity_info(net, input_shape, print_per_layer_stat=False, as_strings=False)

        # KMACs = flops_to_string(GMACs, units="KMac")
        # KParams = params_to_string(params, units="K")

        print(f"\t  MACs: {MACs}")
        print(f"\tParams: {params}")
        if MACs == 0:
            MACs = 1
        if params == 0:
            params = 1

        MAC_l.append(MACs)
        PARAM_l.append(params)

    p1 = softmax(np.array(MAC_l) / 1e6)
    p2 = softmax(np.array(PARAM_l) / 1e6)

    for e in (p1 + p2) / 2:
        print("{:.8f}".format(e * 10))
