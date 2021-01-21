import logging
import pickle
from datetime import timezone, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import setup_seed, ensure_folder

from datasets.tools import int_list_stat, random_negative_method, popular_negative_method, logger_endl
from utils.logger import setup_logger

# top level setting
SEED = 42

# threshold setting
MIN_SESSION_LENGTH = 3
MIN_ITEM_THRESHOLD = 5

# negative sampling setting
NEG_SAMPLE_METHOD = "random"  # random | popular
NEG_NUM_SAMPLE = 100

# original data
DATA_FOLDER = "raw-datas/30music"
DATA_FILE = "30music-200ks.csv"

# how to store information and preprocessed data
STORE_FOLDER = "prepare"  # relative to ${DATA_FOLDER} by default
LOGGER_FILE_NAME = "description.log"
DUMPED_DATASET_FILE = "dataset-len{}-num{}.pkl".format(MIN_SESSION_LENGTH, MIN_ITEM_THRESHOLD)
DUMPED_NEG_SAMPLE_FILE = "neg-{}-n{}.pkl".format(NEG_SAMPLE_METHOD, NEG_NUM_SAMPLE)

# set a small number of rows for easy debugging, set `None` by default
N_ROW_LIMITATION_FOR_DEBUG = None


def run_all(root=DATA_FOLDER, data_file=DATA_FILE, store_path=Path(DATA_FOLDER).joinpath(STORE_FOLDER)):
    setup_seed(SEED, setup_pytorch=False)
    ensure_folder(store_path)
    setup_logger(store_path, LOGGER_FILE_NAME, console_output=True)

    # load data
    file_path = Path(root).joinpath(data_file)
    data = load_data(file_path)
    logging.info(">>>>> Summary of original dataset.")
    print_info(data)
    logger_endl()

    # wash data
    data = filter_data(data)
    logging.info(">>>>> Summary of filtered dataset.")
    print_info(data)
    logger_endl()

    # index -> make session -> gen neg sample
    data, user_to_idx, item_to_idx = make_index(data)
    logger_endl()

    user_to_seqs, user_to_pos_items = gen_session_sample(data)
    logger_endl()

    user_to_neg_items = gen_neg_sample(user_to_seqs, user_to_pos_items, set(item_to_idx.values()))
    logger_endl()

    # save all
    save_data(user_to_seqs, user_to_pos_items, user_to_idx, item_to_idx, store_path)
    save_neg_data(user_to_neg_items, store_path)


def print_info(data):
    logging.info("#User: {}, #Item: {}".format(data.user_id.nunique(), data.item_id.nunique()))

    session_size = data.groupby("session_id").size()
    logging.info("#Session: {}, {}".format(data.session_id.nunique(), int_list_stat(session_size)))

    data_from = datetime.fromtimestamp(data.time.min(), timezone.utc)
    data_to = datetime.fromtimestamp(data.time.max(), timezone.utc)
    logging.info("Time: from {}, to {}".format(data_from, data_to))


def load_data(file_path):
    #   UserId	SessionId	ItemId	Time	ArtistId
    #   27063	1889046	2691760	1402997433	337496
    #   27063	1889046	2691717	1402997784	337496
    #   27063	1889046	2691636	1402998064	337496
    #   27063	1889046	2691702	1402998298	337496
    #   27063	1889046	2691783	1402998576	337496
    logging.info(">>>>> Load data from `{}`".format(file_path))
    data = pd.read_csv(file_path, sep="\t", usecols=[0, 1, 2, 3], nrows=N_ROW_LIMITATION_FOR_DEBUG)
    data.columns = ["user_id", "session_id", "item_id", "time"]

    return data


def filter_data(data, min_item_threshold=MIN_ITEM_THRESHOLD, min_session_length=MIN_SESSION_LENGTH):
    logging.info(
        ">>>>> Filter data with min_item_threshold={}, min_session_length={}".format(
            min_item_threshold, min_session_length
        )
    )
    session_size = data.groupby("session_id").size()
    data = data[np.in1d(data.session_id, session_size[session_size > 1].index)]

    item_freq = data.groupby("item_id").size()
    data = data[np.in1d(data.item_id, item_freq[item_freq >= min_item_threshold].index)]

    session_size = data.groupby("session_id").size()
    data = data[np.in1d(data.session_id, session_size[session_size >= min_session_length].index)]

    return data


def make_index(data):
    logging.info(">>>>> Transform user and item to number.")
    user2idx = {user: (i + 1) for i, user in enumerate(set(data.user_id))}
    item2idx = {item: (i + 1) for i, item in enumerate(set(data.item_id))}
    data.user_id = data.user_id.map(user2idx)
    data.item_id = data.item_id.map(item2idx)

    return data, user2idx, item2idx


def gen_session_sample(data):
    logging.info(">>>>> Generate users' feedback sequences.")

    user_to_seqs = {}
    user_to_pos_items = {}

    for user_id, user_data in tqdm(data.groupby("user_id")):
        seqs_list = []
        all_item_list = []

        for sess_id, sess_data in user_data.groupby("session_id"):
            sort_data = sess_data.sort_values(by="time")
            item_list = sort_data.item_id.to_list()
            seqs_list.append(item_list)
            all_item_list.extend(item_list)

        user_to_seqs[user_id] = seqs_list
        user_to_pos_items[user_id] = set(all_item_list)

    logging.info("Number of session per user: {}".format(int_list_stat([len(a) for a in user_to_seqs.values()])))

    logging.info(
        "Number of item per user had contact with: {}".format(
            int_list_stat([len(a) for a in user_to_pos_items.values()])
        )
    )

    return user_to_seqs, user_to_pos_items


def gen_neg_sample(
    user_to_seqs, user_to_pos_items, total_item_list, sample_method=NEG_SAMPLE_METHOD, num_sample=NEG_NUM_SAMPLE
):
    logging.info(">>>>> Generate negative samples, method={}, num_sample={}".format(sample_method, num_sample))
    if sample_method == "random":
        user_to_neg_items = random_negative_method(user_to_seqs, user_to_pos_items, total_item_list, num_sample)
    elif sample_method == "popular":
        user_to_neg_items = popular_negative_method(user_to_seqs, user_to_pos_items, total_item_list, num_sample)
    else:
        raise ValueError("No such negative sampling method `{}`".format(sample_method))

    return user_to_neg_items


def save_data(user_to_seqs, user_to_pos_items, user_to_idx, item_to_idx, save_folder, filename=DUMPED_DATASET_FILE):
    data_filename = Path(save_folder).joinpath(filename)
    with data_filename.open("wb") as f:
        data = {
            "user_to_seqs": user_to_seqs,
            "user_to_pos_items": user_to_pos_items,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
        }
        pickle.dump(data, f)


def save_neg_data(user_to_neg_items, save_folder, filename=DUMPED_NEG_SAMPLE_FILE):
    neg_data_filename = Path(save_folder).joinpath(filename)
    with neg_data_filename.open("wb") as f:
        pickle.dump(user_to_neg_items, f)


if __name__ == "__main__":
    run_all()
