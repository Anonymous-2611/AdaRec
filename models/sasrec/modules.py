import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, dropout=0.1):
        super().__init__()
        assert num_hidden % num_heads == 0

        self.d_pre_h = num_hidden // num_heads
        self.num_hidden = num_hidden
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)

        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)  # get device later on

    def forward(self, query, key, value):
        # [B, L, C]
        batch_size = query.size(0)

        query_, key_, value_ = [
            #  [B, L, C] => [B, L, C] => [B, L, H, C'] => [B, H, L, C']
            layer(x).view(batch_size, -1, self.num_heads, self.d_pre_h).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        # [B, H, L, C'] * [B, H, C', L] => [B, H, L, L]
        output = torch.matmul(query_, key_.transpose(-2, -1))
        output = output / math.sqrt(query_.size(-1))  # [B, H, L, L]

        # key masking
        key_mask = torch.sign(torch.sum(torch.abs(key), dim=-1))  # [B, L]
        key_mask = key_mask.reshape((batch_size, 1, 1, -1))  # [B, 1, 1, L]
        key_mask = key_mask.repeat((1, self.num_heads, query.size(1), 1))  # [B, H, L, L]
        output = output.masked_fill(key_mask == 0, -1e9)  # [B, H, L, L]
        # output = torch.masked_fill(output, key_mask == 0, -1e9)

        # future blinding / timeline masking
        shape = torch.ones(output.size(-2), output.size(-1))  # [L, L]
        time_mask = torch.tril(shape, diagonal=0, out=None).to(self.dummy.device)  # [L, L]
        output = output.masked_fill(time_mask == 0, -1e9)  # [B, H, L, L]
        # output = torch.masked_fill(output, time_mask == 0, -1e9)

        # Attention
        output = F.softmax(output, dim=-1)  # [B, H, L, L]

        # query masking
        query_mask = torch.sign(torch.sum(torch.abs(query), dim=-1))  # [B, L]
        query_mask = query_mask.reshape((batch_size, 1, -1, 1))  # [B, 1, L, 1]
        query_mask = query_mask.repeat((1, self.num_heads, 1, key.size(1)))  # [B, H, L, L]
        output = output * query_mask  # [B, H, L, L]
        # do NOT use inplace op `output *= query_mask`, this may cause error when back-prop to `F.softmax` above

        output = self.dropout(output)  # [B, H, L, L]
        output = torch.matmul(output, value_)  # [B, H, L, L] * [B, H, L, C'] => [B, H, L, C']
        output = output.transpose(1, 2).contiguous()  # [B, L, H, C']
        output = output.view(batch_size, -1, self.num_hidden)  # [B, L, C]

        output += query
        return output  # [B, L, C]


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, dropout):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(num_hidden, num_hidden, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout)

        self.conv2 = torch.nn.Conv1d(num_hidden, num_hidden, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout)

    def forward(self, x_in):
        # x_in : [B, L, C]
        x = x_in.transpose(-1, -2)  # [B, C, L]

        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(self.conv2(x))

        x = x.transpose(-1, -2)
        x += x_in  # [B, C, L]
        return x


class SASBlock(nn.Module):
    def __init__(self, num_hidden, num_heads, dropout, use_eps):
        super(SASBlock, self).__init__()

        self.ln1 = nn.LayerNorm(num_hidden, eps=1e-8)
        self.attn = MultiHeadAttention(num_hidden, num_heads, dropout)

        self.ln2 = nn.LayerNorm(num_hidden, eps=1e-8)
        self.feedforward = PointWiseFeedForward(num_hidden, dropout)

        if use_eps:
            self.eps = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.eps = 1.0

    def forward(self, x_in):
        x = self.ln1(x_in)
        x = self.attn(x, x, x)
        x = self.ln2(x)
        x = self.feedforward(x)
        return x_in + self.eps * x  # [B, L, C]
