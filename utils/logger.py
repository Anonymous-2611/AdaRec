import logging
from pathlib import Path


def setup_logger(folder_path, log_file_name="logger.log", console_output=False):
    dir_root = Path(folder_path)
    full_path = dir_root.joinpath(log_file_name)
    print("File: ", full_path)

    already_exist = Path(full_path).exists()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s| %(message)s", "%m-%d|%H:%M:%S")

    file_hdl = logging.FileHandler(full_path)
    file_hdl.setFormatter(formatter)

    root_logger.addHandler(file_hdl)

    if console_output:
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(formatter)
        root_logger.addHandler(console_hdl)

    if already_exist:
        logging.info("")
        logging.info("")
        logging.info(f">>>>> Logger file {full_path} already exist, append to it. <<<<<")
        logging.info("")
        logging.info("")


def setup_simple_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s| %(message)s", "%m-%d|%H:%M:%S")

    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(formatter)
    root_logger.addHandler(console_hdl)
