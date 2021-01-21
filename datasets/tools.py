import logging
from collections import Counter

import numpy as np
from tqdm import tqdm


def logger_endl():
    logging.info("")


def int_list_stat(data, indent=""):
    _min, _max = np.min(data), np.max(data)
    _avg, _std = np.mean(data), np.std(data)

    return "{}min={}, max={}, avg={:.4f}, std={:.4f}".format(indent, _min, _max, _avg, _std)


def random_negative_method(user_to_seqs, user_to_pos_items, total_item_set, num_sample):
    user_to_neg_items = {}
    for user_id, pos_items in tqdm(user_to_pos_items.items()):
        neg_items_candidates = list(total_item_set - pos_items)
        user_to_neg_items[user_id] = np.random.choice(neg_items_candidates, num_sample, replace=False).tolist()
    return user_to_neg_items


def popular_negative_method(user_to_seqs, user_to_pos_items, total_item_set, num_sample):
    item_counter = Counter()
    for seqs in user_to_seqs.values():
        merge_seq = sum(seqs, [])
        item_counter.update(merge_seq)
    popular_items = [k for k, v in item_counter.most_common()]

    user_to_neg_items = {}
    for user_id, pos_items in tqdm(user_to_pos_items.items()):
        neg_items = []
        for item in popular_items:
            if len(neg_items) == num_sample:
                break
            if item in pos_items:
                continue
            neg_items.append(item)
        user_to_neg_items[user_id] = neg_items

    return user_to_neg_items
