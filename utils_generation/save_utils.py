import os
import time
from typing import Optional

import numpy as np
import pandas as pd


def getDir(dataset_name_w_num, args):
    d = "{}_{}_{}_{}".format(args.model, dataset_name_w_num, args.prefix, args.token_place)
    if args.tag != "":
        d += "_{}".format(args.tag)

    return os.path.join(args.save_base_dir, d)


def maybeAppendProjectSuffix(method, project_along_mean_diff):
    if project_along_mean_diff:
        return method + "-md"
    return method


def make_params_filename(model, prefix, method, train_set, seed):
    return "{}_{}_{}_{}_{}_{}".format(
        model, prefix, method, "all", train_set, seed)


def get_probs_save_dir_path(root_dir, model, method, project_along_mean_diff, seed, train_set):
    """Returns the path to the directory where the classifier probabilities are saved."""
    return f"{root_dir}/states_{model}_{maybeAppendProjectSuffix(method, project_along_mean_diff)}_{seed}/{train_set}"


def saveParams(save_dir, name, coef: np.ndarray, intercept: Optional[np.ndarray]):
    path = os.path.join(save_dir, "params")
    np.save(os.path.join(path, "coef_{}.npy".format(name)), coef)
    if intercept is not None:
        np.save(os.path.join(path, "intercept_{}.npy".format(name)), intercept)


def saveFrame(frame_dict, args):
    if not os.path.exists(args.save_base_dir):
        os.makedirs(args.save_base_dir, exist_ok=True)
    for key, frame in frame_dict.items():

        directory = getDir(key, args)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        frame.to_csv(os.path.join(directory, "frame.csv"), index = False)

    print("Successfully saving datasets to each directory.")


def saveArray(array_list, typ_list, key, args):
    directory = getDir(key, args)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # hidden states is num_data * layers * dim
    # logits is num_data * vocab_size
    for (typ, array) in zip(typ_list, array_list):
        if args.save_all_layers or "logits" in typ:
            np.save(os.path.join(directory, "{}.npy".format(typ)), array)
        else:
            # only save the last layers for encoder hidden states
            for idx in args.states_index:
                np.save(os.path.join(directory, "{}_{}{}.npy".format(typ, args.states_location, idx)),
                        array[:, idx,:])




def saveRecords(records, args):
    f = os.path.join(args.save_base_dir, "{}.csv".format(args.save_csv_name))
    if not os.path.exists(f):
        csv = pd.DataFrame(columns = ["time", "model", "dataset", "prompt_idx", "num_data", "population", "prefix", "cal_zeroshot", "cal_hiddenstates", "log_probs", "calibrated", "tag"])
    else:
        csv = pd.read_csv(f)

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for dic in records:
        dic["time"] = t
        spliter = dic["dataset"].split("_")
        dic["dataset"], dic["prompt_idx"] = spliter[0], int(spliter[2][6:])
    csv = csv.append(records, ignore_index = True)

    csv.to_csv(f, index = False)

    print("Successfully saved {} items in records to {}".format(len(records), f))
