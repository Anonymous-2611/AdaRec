import logging

import numpy as np

from models.nas.modules.operators import OPERATOR_NAME


def dict_to_logger(data, exclude_list=None):
    res = tabular_pretty_print(dict_to_list(data, exclude_list))

    for s in res:
        logging.info(s)


def dict_to_list(data: dict, exclude_list: list):
    res = []
    for k in sorted(data.keys()):
        if k not in exclude_list:
            res.append([str(k), str(data[k])])
    return res


def tabular_pretty_print(grid):
    lens = [max(map(len, col)) for col in zip(*grid)]

    fmt = " | ".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in grid]

    sep = ["~" * len(table[0])]
    table = sep + table + sep

    res = []
    for idx, line in enumerate(table):
        if idx == 0 or idx == len(table) - 1:
            ps = "\t* {} *".format(line)
        else:
            ps = "\t| {} |".format(line)
        res.append(ps)
    return res


def output_alpha_to_logger(arch):
    logging.info("-*" * 20)
    for idx, edges_into_a_node in enumerate(arch):
        logging.info(f"To Node-{idx + 1}")
        for src, edge in enumerate(edges_into_a_node):
            edge_idx = np.argmax(edge)
            edge_name = OPERATOR_NAME[edge_idx]
            if src == 0:
                from_name = ".Input"
            else:
                from_name = f"Node-{src}"
            logging.info(f"\tfrom {from_name} -> {edge_name}")
    logging.info("-*" * 20)
