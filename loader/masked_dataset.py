import torch
from torch.utils.data import Dataset

from loader.funcs import ensure_size


class MaskedTrainDataset(Dataset):
    def __init__(self, user_to_seqs, max_len, mask_prob, token_mask, token_pad, num_items, rng):
        # user_with_items: dict, {user_id: [[item_id,...], [item_id,...]]}

        self.user_to_seqs = user_to_seqs

        self.max_len = max_len
        self.mask_prob = mask_prob

        self.TOKEN_MASK = token_mask
        self.TOKEN_PAD = token_pad

        self.num_items = num_items
        self.rng = rng

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

        training_seq = self.user_to_seqs[user][idx]

        tokens = []
        labels = []
        for s in training_seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.TOKEN_MASK)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(self.TOKEN_PAD)

        tokens = ensure_size(tokens, self.max_len, self.TOKEN_PAD)
        labels = ensure_size(labels, self.max_len, self.TOKEN_PAD)

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class MaskedEvalDataset(Dataset):
    def __init__(self, user_to_seqs, user_to_answers, max_len, token_mask, token_pad, user_to_neg_items):
        self.user_to_seqs = user_to_seqs

        self.max_len = max_len

        self.TOKEN_MASK = token_mask
        self.TOKEN_PAD = token_pad

        self.user_to_answers = user_to_answers

        self.user_to_neg_items = user_to_neg_items

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
        answer = self.user_to_answers[user][idx]
        negs = self.user_to_neg_items[user]

        candidates = answer + negs

        seq = seq + [self.TOKEN_MASK]
        seq = ensure_size(seq, self.max_len, self.TOKEN_PAD)

        return torch.tensor(seq, dtype=torch.long), torch.tensor(candidates, dtype=torch.long)
