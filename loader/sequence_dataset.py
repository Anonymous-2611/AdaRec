import torch
from torch.utils.data import Dataset

from loader.funcs import ensure_size


class SequenceTrainDataset(Dataset):
    def __init__(self, user_to_seqs, max_len, token_pad):
        # user_with_items: dict, {user_id: [[item_id,...], [item_id,...]]}

        self.user_to_seqs = user_to_seqs
        self.max_len = max_len
        self.TOKEN_PAD = token_pad

        self.idx_to_sample = {}  # {index: (u, idx)}

        index = 0
        for u, seqs in self.user_to_seqs.items():
            for idx in range(len(seqs)):
                self.idx_to_sample[index] = (u, idx)
                index += 1
        self.fast_len = index

    def __len__(self):
        return self.fast_len

    def __getitem__(self, index):
        user, idx = self.idx_to_sample[index]

        seq = self.user_to_seqs[user][idx]

        pre = ensure_size(seq[:-1], self.max_len, self.TOKEN_PAD)
        post = ensure_size(seq[1:], self.max_len, self.TOKEN_PAD)

        return torch.tensor(pre, dtype=torch.long), torch.tensor(post, dtype=torch.long)


class SequenceEvalDataset(Dataset):
    def __init__(self, user_to_seqs, user_to_answers, max_len, token_pad, user_to_neg_items):

        self.user_to_seqs = user_to_seqs
        self.user_to_answers = user_to_answers
        self.user_to_neg_items = user_to_neg_items
        self.max_len = max_len
        self.TOKEN_PAD = token_pad

        self.idx_to_sample = {}  # {index: (u, idx)}

        index = 0
        for u, seqs in self.user_to_seqs.items():
            for idx in range(len(seqs)):
                self.idx_to_sample[index] = (u, idx)
                index += 1
        self.fast_len = index

    def __len__(self):
        return self.fast_len

    def __getitem__(self, index):
        user, idx = self.idx_to_sample[index]

        seq = self.user_to_seqs[user][idx]
        seq = ensure_size(seq, self.max_len, self.TOKEN_PAD)

        answer = self.user_to_answers[user][idx]
        negs = self.user_to_neg_items[user]
        candidates = answer + negs

        return torch.tensor(seq, dtype=torch.long), torch.tensor(candidates, dtype=torch.long)
