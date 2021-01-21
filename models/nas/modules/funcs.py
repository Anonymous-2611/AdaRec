import torch


def get_logits(z, eps=1e-8):
    shifted_z = z - torch.max(z, dim=-1, keepdim=True)[0]
    normalizer = torch.log(torch.sum(torch.exp(shifted_z)) + eps)
    return shifted_z - normalizer


def get_gumbel_noise(shape, eps=1e-8):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)
