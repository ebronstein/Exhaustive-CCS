import argparse
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from utils.file_utils import get_model_short_name


def get_hidden_states_dir(dataset_name_w_num: str, args: argparse.Namespace):
    """Return the directory where hidden states are saved.

    Args:
        dataset_name_w_num (str): dataset name with number of examples
            (e.g., "imdb_1000").
        args (argparse.Namespace): CLI arguments.
    """
    model_short_name = get_model_short_name(args.model)
    d = "{}_{}_{}_{}".format(
        model_short_name, dataset_name_w_num, args.prefix, args.token_place
    )
    if args.tag != "":
        d += "_{}".format(args.tag)

    return os.path.join(args.save_base_dir, d)


def saveArray(array_list, typ_list, key, args: argparse.Namespace):
    directory = get_hidden_states_dir(key, args)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # hidden states is num_data * layers * dim
    # logits is num_data * vocab_size
    for typ, array in zip(typ_list, array_list):
        if args.save_all_layers or "logits" in typ:
            np.save(os.path.join(directory, "{}.npy".format(typ)), array)
        else:
            # only save the last layers for encoder hidden states
            for idx in args.states_index:
                np.save(
                    os.path.join(
                        directory,
                        "{}_{}{}.npy".format(typ, args.states_location, idx),
                    ),
                    array[:, idx, :],
                )


def saveRecords(records, args: argparse.Namespace):
    f = os.path.join(args.save_base_dir, "{}.csv".format(args.save_csv_name))
    if not os.path.exists(f):
        csv = pd.DataFrame(
            columns=[
                "time",
                "model",
                "dataset",
                "prompt_idx",
                "num_data",
                "population",
                "prefix",
                "cal_zeroshot",
                "cal_hiddenstates",
                "log_probs",
                "calibrated",
                "tag",
            ]
        )
    else:
        csv = pd.read_csv(f)

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for dic in records:
        dic["time"] = t
        spliter = dic["dataset"].split("_")
        dic["dataset"], dic["prompt_idx"] = spliter[0], int(spliter[2][6:])
    csv = csv.append(records, ignore_index=True)

    csv.to_csv(f, index=False)

    print(
        "Successfully saved {} items in records to {}".format(len(records), f)
    )
