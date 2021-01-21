import pickle
from pathlib import Path
from math import ceil

from torch.utils.data import DataLoader

from loader.sequence_dataset import SequenceEvalDataset, SequenceTrainDataset
from utils.others import flatten_2d


class SequenceLoaderProvider:
    def __init__(self, args):
        # self.args = args
        self.train_batch_size = args.loader_train_batch_size
        self.val_batch_size = args.loader_val_batch_size
        self.test_batch_size = args.loader_test_batch_size
        self.max_len = args.loader_max_len
        self.gen_sub_session = args.loader_generate_sub_session

        self.user_to_seqs = None
        self.user_to_neg_items = None

        self.num_item = None
        self.num_user = None

        self.TOKEN_PAD = 0

        self._init(args.data_folder, args.data_main, args.data_neg)

    def _init(self, folder_path, main_file_name, neg_file_name):
        folder = Path(folder_path)
        main_file = folder.joinpath(main_file_name)
        neg_file = folder.joinpath(neg_file_name)

        with main_file.open("rb") as f1:
            main_data = pickle.load(f1)
            self.user_to_seqs = main_data["user_to_seqs"]
            self.num_user = len(main_data["user_to_idx"])

            item_to_idx = main_data["item_to_idx"]
            self.num_item = len(item_to_idx)

            # Make sure idx `0` is reserved for TOKEN_PAD.
            assert 0 not in item_to_idx, "Invalid preprocessed data file, index `0` shouldn't have been here."

        with neg_file.open("rb") as f2:
            self.user_to_neg_items = pickle.load(f2)

        self._check_data()
        self._do_slice()

    def _check_data(self):
        # make sure all negative item lists are of the same length
        len_first = None
        for u, items in self.user_to_neg_items.items():
            if len_first is None:
                len_first = len(items)
            assert len(items) == len_first, "different length of negative item list at user {}".format(u)

    def _do_slice(self):
        # Make sure `[train,[...]], valid, test` is satisfied.
        # Note that [train,...] must more than two elements.
        #
        # For self-regression model, [0:n-1] elements are training
        # feed-in data and [1:n] elements are training target data.

        min_len_threshold = 4

        new_data = {}
        for u, seqs in self.user_to_seqs.items():  # for every users
            new_data[u] = []
            for _seq in seqs:  # for every seqs
                size = len(_seq)
                num_parts = ceil(size / self.max_len)
                for idx in range(num_parts):  # for every slices
                    f = max(0, size - (idx + 1) * self.max_len)
                    t = size - idx * self.max_len
                    if t - f >= min_len_threshold:
                        new_data[u].append(_seq[f:t])
        self.user_to_seqs = new_data

    def report_info(self):
        sess_stat = flatten_2d([[len(ee) for ee in e] for e in self.user_to_seqs.values()])
        s = "Data: "
        s += "#User:{}".format(self.num_user)
        s += " #Item:{}".format(self.num_item)
        s += " #Action:{}".format(sum(sess_stat))
        s += " #Sess:{}".format(len(sess_stat))
        s += " (.avg={:.3f} .min={} .max={})".format(sum(sess_stat) / len(sess_stat), min(sess_stat), max(sess_stat))

        return s

    def get_loader(self):
        return self.training_loader(), self.validation_loader(), self.test_loader()

    @staticmethod
    def generate_sub_session(user_to_seqs):
        min_size = 2
        new_data = {}
        for u, seqs in user_to_seqs.items():  # for every users
            new_data[u] = []
            for _seq in seqs:  # for every seqs
                # a b c d e
                if len(_seq) <= min_size:
                    new_data[u].append(_seq)
                    continue
                for end_idx in range(min_size, len(_seq) + 1):
                    slice_seq = _seq[0:end_idx]
                    new_data[u].append(slice_seq)
        return new_data

    def training_loader(self):
        user_to_seqs = {u: [seq[:-2] for seq in seqs] for u, seqs in self.user_to_seqs.items()}
        if self.gen_sub_session:
            user_to_seqs = self.generate_sub_session(user_to_seqs)
        dataset = SequenceTrainDataset(user_to_seqs, self.max_len, self.TOKEN_PAD)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)
        return loader

    def validation_loader(self):
        # use penultimate as target
        user_to_seqs = {u: [seq[:-2] for seq in seqs] for u, seqs in self.user_to_seqs.items()}
        val_answer = {u: [seq[-2:-1] for seq in seqs] for u, seqs in self.user_to_seqs.items()}
        dataset = SequenceEvalDataset(user_to_seqs, val_answer, self.max_len, self.TOKEN_PAD, self.user_to_neg_items)
        loader = DataLoader(dataset, batch_size=self.val_batch_size, shuffle=False)
        return loader

    def test_loader(self):
        # use the last as target
        user_to_seqs = {u: [seq[:-2] for seq in seqs] for u, seqs in self.user_to_seqs.items()}
        test_answer = {u: [seq[-1:] for seq in seqs] for u, seqs in self.user_to_seqs.items()}
        dataset = SequenceEvalDataset(user_to_seqs, test_answer, self.max_len, self.TOKEN_PAD, self.user_to_neg_items)
        loader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False)
        return loader

    @property
    def num_token(self):
        return self.num_item + 1

    @property
    def token_pad(self):
        return self.TOKEN_PAD

