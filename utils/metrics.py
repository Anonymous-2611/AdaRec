import logging

import numpy as np

from utils.printer import tabular_pretty_print


def sample_top_k_old(a, top_k):
    idx = np.argsort(a)[:, ::-1]
    idx = idx[:, :top_k]
    return idx


def sample_top_k(a, top_k):
    idx = np.argpartition(a, -top_k)[:, -top_k:]
    part = np.take_along_axis(a, idx, 1)
    return np.take_along_axis(idx, np.argsort(part), 1)[:, ::-1]


def sample_top_ks_old(a, top_ks):
    # O(n * log(n)) + b * O(1)
    idx = np.argsort(a)[:, ::-1]
    for k in top_ks:
        yield idx[:, :k]


def sample_top_ks(a, top_ks):
    # O(b * (n + k * log(k)))
    for k in top_ks:
        idx = np.argpartition(a, -k)[:, -k:]
        part = np.take_along_axis(a, idx, 1)
        yield np.take_along_axis(idx, np.argsort(part), 1)[:, ::-1]


# mrr@K, hit@K, ndcg@k
def get_metric(rank_indices):
    mrr_list, hr_list, ndcg_list = [], [], []
    for t in rank_indices:
        if len(t):
            mrr_list.append(1.0 / (t[0][0] + 1))
            ndcg_list.append(1.0 / np.log2(t[0][0] + 2))
            hr_list.append(1.0)
        else:
            mrr_list.append(0.0)
            ndcg_list.append(0.0)
            hr_list.append(0.0)

    return mrr_list, hr_list, ndcg_list


class SRSMetric:
    def __init__(self, k_list, use_mrr=True, use_hit=True, use_ndcg=True):
        self.k_list = k_list

        self.mrr_list, self.use_mrr = None, use_mrr
        self.hit_list, self.use_hit = None, use_hit
        self.ndcg_list, self.use_ndcg = None, use_ndcg

        self.mrr = None
        self.hit = None
        self.ndcg = None

    def setup_and_clean(self):
        if self.use_mrr:
            self.mrr = {}
            self.mrr_list = {}
            self._setup_one(self.mrr_list)
        if self.use_hit:
            self.hit = {}
            self.hit_list = {}
            self._setup_one(self.hit_list)
        if self.use_ndcg:
            self.ndcg = {}
            self.ndcg_list = {}
            self._setup_one(self.ndcg_list)

    def _setup_one(self, obj):
        for k in self.k_list:
            obj[k] = []

    @staticmethod
    def _get_idx(argsort_res, real_idx):
        equ_array = argsort_res == real_idx
        row_idx = np.argmax(equ_array, 1)
        row_idx[np.any(equ_array, 1)] += 1

        return row_idx

    def submit(self, predict_probs, real_idx):
        # predict_probs [B, num_items]
        # real_idx      [B, 1]
        predict_probs = np.array(predict_probs)
        for raw, k in zip(sample_top_ks(predict_probs, self.k_list), self.k_list):
            row_idx = self._get_idx(raw, real_idx)

            mrr_list, hit_list, ndcg_list = [], [], []
            for t in row_idx:
                if t:
                    mrr_list.append(1.0 / t)
                    ndcg_list.append(1.0 / np.log2(t + 1))
                    hit_list.append(1.0)
                else:
                    mrr_list.append(0.0)
                    ndcg_list.append(0.0)
                    hit_list.append(0.0)

            if self.use_mrr:
                self.mrr_list[k].extend(mrr_list)
            if self.use_hit:
                self.hit_list[k].extend(hit_list)
            if self.use_ndcg:
                self.ndcg_list[k].extend(ndcg_list)

    def calc(self):
        if self.use_mrr:
            self._calc_one(self.mrr, self.mrr_list)
        if self.use_hit:
            self._calc_one(self.hit, self.hit_list)
        if self.use_ndcg:
            self._calc_one(self.ndcg, self.ndcg_list)

    def _calc_one(self, score_dict, metric_list):
        for k in self.k_list:
            score_dict[k] = np.mean(metric_list[k])

    def output_to_logger(self, decimal=4):
        def fmt_f(num, d):
            fmt_string = "{{:.{}f}}".format(d)
            return fmt_string.format(num)

        content = [[""]]
        if self.use_mrr:
            content[0].append("MRR")
        if self.use_hit:
            content[0].append("HIT")
        if self.use_ndcg:
            content[0].append("NDCG")

        for k in self.k_list:
            line = ["K={}".format(k)]
            if self.use_mrr:
                line.append(fmt_f(self.mrr[k], decimal))
            if self.use_hit:
                line.append(fmt_f(self.hit[k], decimal))
            if self.use_ndcg:
                line.append(fmt_f(self.ndcg[k], decimal))
            content.append(line)

        lines = tabular_pretty_print(content)
        for line in lines:
            logging.info(line)





class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string="{}"):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string="{}"):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string="{}"):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, fmt):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=fmt)


if __name__ == "__main__":
    from utils.logger import setup_simple_logger

    setup_simple_logger()

    tool = SRSMetric(k_list=[1, 2, 3, 4])
    tool.setup_and_clean()
    """
        0   1   2   3   4   5   6
        0.1 0.2 0.1 0.7 0.1 0.3 0.5  = 3
        0.2 0.3 0.4 0.2 0.7 0.8 0.1  = 5
        """
    tool.submit(np.array([[0.1, 0.2, 0.1, 0.7, 0.1, 0.3, 0.5]]), [[3]])
    tool.submit(np.array([[0.2, 0.3, 0.4, 0.2, 0.7, 0.8, 0.1]]), [[4]])
    tool.calc()

    tool.output_to_logger()

    cc = ClsMetric(num_cls=3, use_cm=True)
    cc.setup_and_clean()
    pred = [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]
    real = [[0], [1], [2], [0], [1], [2]]
    cc.submit(pred[2:4], real[2:4])
    cc.submit(pred[0:2], real[0:2])
    cc.submit(pred[4:], real[4:])
    # cc.submit(pred, real)
    cc.calc()
    cc.output_to_logger(layout="horizontal")
