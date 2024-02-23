import argparse
import collections
import json
import os
import random
import time
import typing
import warnings
from typing import Any, Iterable, Literal, Optional, Union

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.config.custom_containers import DogmaticDict, DogmaticList
from sacred.observers import FileStorageObserver

from utils.file_utils import get_model_short_name
from utils_extraction import load_utils
from utils_extraction.load_utils import (
    get_params_dir,
    load_hidden_states_for_datasets,
    make_permutation_dict,
    maybe_append_project_suffix,
    save_params,
)
from utils_extraction.method_utils import is_method_unsupervised, mainResults
from utils_generation import hf_utils
from utils_generation import parser as parser_utils
from utils_generation.hf_auth_token import HF_AUTH_TOKEN

ALL_DATASETS = [
    "imdb",
    "amazon-polarity",
    "ag-news",
    "dbpedia-14",
    "copa",
    "rte",
    "boolq",
    "qnli",
    "piqa",
]

PrefixType = Literal[
    "normal",
    "confusion",
    "confusion2",
    "confusion3",
    "confusion4",
    "confusion6",
    "confusion7",
    "iamincorrect",
    "dadnotrust",
    "dadisincorrect",
    "teachernoimitate",
    "normal-dot",
    "normal-mark",
    "normal-thatsright",
]

MethodType = Literal["0-shot", "TPC", "KMeans", "LR", "BSS", "CCS"]

ex = Experiment()


def methodHasLoss(method):
    return method in ["BSS", "CCS"] or method.startswith("RCCS")


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])


@ex.config
def sacred_config():
    # Experiment name
    name: str = "extraction"
    model = "/scratch/data/meta-llama/Llama-2-7b-chat-hf"
    datasets: Union[str, list[str]] = "imdb"
    prefix: PrefixType = "normal"
    data_num: int = 1000
    method_list: Union[MethodType, list[MethodType]] = "CCS"
    mode: Literal["auto", "minus", "concat"] = "auto"
    save_dir = "extraction_results"
    load_dir = "generation_results"
    load_classifier: bool = False
    # Directory where the saved method parameters will be loaded from.
    # The parameters are expected to be saved in `params_load_dir/bias.npy` and
    # `params_load_dir/intercept.npy`.
    params_load_dir: Optional[str] = None
    location: Literal["auto", "encoder", "decoder"] = "auto"
    layer: int = -1
    num_layers = hf_utils.get_num_hidden_layers(model)
    # File name where zero-shot results will be saved.
    zero: str = "zero_shot"
    seed: int = 0
    save_states: bool = True
    save_params = True
    save_results: bool = True
    test_on_train: bool = False
    project_along_mean_diff: bool = False
    verbose: bool = False

    # TODO: delete these unused config parameters
    test: Literal["testone", "testall"] = "testall"
    append: bool = False
    overwrite: bool = False

    exp_dir = load_utils.get_exp_dir(save_dir, name, model, datasets, seed)
    ex.observers.append(FileStorageObserver(exp_dir, copy_sources=False))


def _convert_dogmatics_to_standard(obj: Any) -> Any:
    """Recursively converts an object with Sacred Dogmatics to a standard Python object."""
    if isinstance(obj, DogmaticDict):
        return {k: _convert_dogmatics_to_standard(v) for k, v in obj.items()}
    elif isinstance(obj, DogmaticList):
        return [_convert_dogmatics_to_standard(elem) for elem in obj]
    elif isinstance(obj, collections.abc.Mapping):
        return {k: _convert_dogmatics_to_standard(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        # Exclude strings as they are also iterable but should not be treated as a list of characters here.
        return [_convert_dogmatics_to_standard(elem) for elem in obj]
    else:
        return obj


def _format_config(config: dict) -> dict:
    config = _convert_dogmatics_to_standard(config)

    # Convert single strings to lists.
    for key in ["datasets", "method_list"]:
        if isinstance(config[key], str):
            config[key] = [config[key]]

    if config["prefix"] not in typing.get_args(PrefixType):
        raise ValueError(f"Invalid prefix: {config['prefix']}")

    config["location"] = parser_utils.get_states_location_str(
        config["location"], config["model"], use_auth_token=HF_AUTH_TOKEN
    )
    # TODO: validate location more extensively.
    if config["location"] == "decoder" and config["layer"] < 0:
        config["layer"] += config["num_layers"]

    if config["test"] == "testone":
        raise ValueError(
            "Current extraction program does not support applying method on prompt-specific level. Set --test=testall."
        )

    return config


@ex.automain
def main(
    model, save_dir, exp_dir, _config: dict, seed: int, _seed: int, _log, _run
):
    _config = _format_config(_config)
    dataset_list = _config["datasets"]

    run_id = _run._id
    run_dir = os.path.join(exp_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prefix = _config["prefix"]
    _log.info(
        "---------------- model = {}, prefix = {} ----------------".format(
            model, prefix
        )
    )

    # TODO: look into how zero-shot results are being saved.
    if "0-shot" in _config["method_list"]:
        raise NotImplementedError(
            "Zero-shot extraction is not yet implemented."
        )
        # # load zero-shot performance
        # rawzeros = pd.read_csv(
        #     os.path.join(_config["load_dir"], "{}.csv".format(_config["zero"]))
        # )
        # # Get the global zero acc dict (setname, [acc])
        # zeros_acc = get_zeros_acc(
        #     _config["load_dir"],
        #     csv_name=_config["zero"],
        #     mdl_name=model,
        #     dataset_list=dataset_list,
        #     prefix=prefix,
        # )
        # for setname in dataset_list:
        #     if _config["prompt_save_level"] == "all":
        #         eval_csv = eval_adder(
        #             eval_csv,
        #             model,
        #             prefix,
        #             "0-shot",
        #             "",
        #             "",
        #             setname,
        #             np.mean(zeros_acc[setname]),
        #             np.std(zeros_acc[setname]),
        #             "",
        #             "",
        #             "",
        #             "",
        #         )
        #     else:  # For each prompt, save one line
        #         for idx in range(len(zeros_acc[setname])):
        #             eval_csv = eval_adder(
        #                 eval_csv,
        #                 model,
        #                 prefix,
        #                 "0-shot",
        #                 prompt_level=idx,
        #                 train="",
        #                 test=setname,
        #                 accuracy=zeros_acc[setname][idx],
        #                 std="",
        #                 ece="",
        #                 location="",
        #                 layer="",
        #                 loss="",
        #                 sim_loss="",
        #                 cons_loss="",
        #             )

        # if _config["save_results"]:
        #     eval_csv.to_csv(eval_results_path, index=False)
        #     if _config["verbose"]:
        #         _log.info(
        #             "Saved zero-shot performance to %s", eval_results_path
        #         )

    train_results = []
    eval_results = collections.defaultdict(list)

    for method in _config["method_list"]:
        if method == "0-shot":
            continue
        _log.info("-------- method = %s --------", method)

        project_along_mean_diff = (
            "-md" in method or _config["project_along_mean_diff"]
        )
        if project_along_mean_diff:
            method = method.replace("-md", "")

        method_use_concat = (method in {"CCS", "Random"}) or method.startswith(
            "RCCS"
        )

        mode = (
            _config["mode"]
            if _config["mode"] != "auto"
            else ("concat" if method_use_concat else "minus")
        )
        # load the data_dict and permutation_dict
        data_dict = load_hidden_states_for_datasets(
            _config["load_dir"],
            mdl_name=model,
            dataset_list=dataset_list,
            prefix=prefix,
            location=_config["location"],
            layer=_config["layer"],
            mode=mode,
            logger=_log,
        )
        permutation_dict = {
            ds: make_permutation_dict(data_dict[ds]) for ds in dataset_list
        }
        assert data_dict.keys() == set(dataset_list)
        assert permutation_dict.keys() == set(dataset_list)

        test_dict = {ds: range(len(data_dict[ds])) for ds in dataset_list}

        train_sets_str = load_utils.get_combined_datasets_str(dataset_list)
        train_list = dataset_list
        projection_dict = {
            key: list(range(len(data_dict[key]))) for key in train_list
        }

        n_components = 1 if method == "TPC" else -1

        method_str = "CCS" if method.startswith("RCCS") else method
        params_dir = get_params_dir(
            run_dir,
            maybe_append_project_suffix(method_str, project_along_mean_diff),
            prefix,
        )

        method_ = method
        constraints = None
        if method.startswith("RCCS"):
            method_ = "CCS"
            if method != "RCCS0":
                # TODO: replace this with call to parameter-loading function.
                constraints = np.load(os.path.join(params_dir, "coef.npy"))
                old_biases = np.load(os.path.join(params_dir, "intercept.npy"))

        load_classifier_dir = (
            _config["params_load_dir"] if _config["load_classifier"] else None
        )

        # return a dict with the same shape as test_dict
        # for each key test_dict[key] is a unitary list
        res, lss, ece, pmodel, cmodel = mainResults(
            data_dict=data_dict,
            permutation_dict=permutation_dict,
            projection_dict=projection_dict,
            test_dict=test_dict,
            projection_method="PCA",
            n_components=n_components,
            load_classifier_dir=load_classifier_dir,
            prefix=prefix,
            classification_method=method_,
            print_more=_config["verbose"],
            save_probs=_config["save_states"],
            test_on_train=_config["test_on_train"],
            constraints=constraints,
            project_along_mean_diff=project_along_mean_diff,
            run_dir=run_dir,
            seed=seed,
            run_id=run_id,
        )

        # save params except for KMeans
        if method in ["TPC", "BSS", "CCS", "Random", "LR"]:
            if method in ["TPC", "BSS"]:
                coef, bias = (
                    cmodel.coef_ @ pmodel.getDirection(),
                    cmodel.intercept_,
                )
            elif method in ["CCS", "Random"]:
                coef = cmodel.coef
                bias = cmodel.bias
            elif method == "LR":
                coef, bias = cmodel.coef_, cmodel.intercept_
            else:
                assert False
            if _config["save_params"]:
                save_params(
                    params_dir,
                    coef,
                    bias,
                )

        if method.startswith("RCCS"):
            coef_and_bias = cmodel.best_theta
            coef = coef_and_bias[:, :-1]
            bias = coef_and_bias[:, -1]
            if method != "RCCS0":
                coef = np.concatenate([constraints, coef], axis=0)
                bias = np.concatenate([old_biases, bias], axis=0)
            if _config["save_params"]:
                save_params(params_dir, coef, bias)

        acc, std, loss, sim_loss, cons_loss = (
            getAvg(res),
            np.mean([np.std(lis) for lis in res.values()]),
            *np.mean([np.mean(lis, axis=0) for lis in lss.values()], axis=0),
        )
        ece_dict = {
            key: [ece[0] for ece in ece_vals] for key, ece_vals in ece.items()
        }
        ece_flip_dict = {
            key: [ece[1] for ece in ece_vals] for key, ece_vals in ece.items()
        }
        mean_ece = getAvg(ece_dict)
        mean_ece_flip = getAvg(ece_flip_dict)
        _log.info(
            "method = {:8}, prompt_level = {:8}, train_set = {:10}, avgacc is {:.2f}, std is {:.2f}, loss is {:.4f}, sim_loss is {:.4f}, cons_loss is {:.4f}, ECE is {:.4f}, ECE (1-p) is {:.4f}".format(
                maybe_append_project_suffix(method, project_along_mean_diff),
                "all",
                train_sets_str,
                100 * acc,
                100 * std,
                loss,
                sim_loss,
                cons_loss,
                mean_ece,
                mean_ece_flip,
            )
        )

        # Organize train and eval results.
        for test_set in dataset_list:
            # TODO: handle the case where consecutive prompts may not be used,
            # so `res` should store the actual prompt indices rather than just
            # a list of the accuracy (same for the other results from
            # `mainResults`).
            for prompt_idx in range(len(res[test_set])):
                loss, sim_loss, cons_loss = (
                    lss[test_set][prompt_idx]
                    if methodHasLoss(method)
                    else ("", "", "")
                )
                train_results.append(
                    {
                        "model": model,
                        "prefix": prefix,
                        "method": maybe_append_project_suffix(
                            method, project_along_mean_diff
                        ),
                        "prompt_level": prompt_idx,
                        "train": train_sets_str,
                        "test": test_set,
                        "location": _config["location"],
                        "layer": _config["layer"],
                        "loss": loss,
                        "sim_loss": sim_loss,
                        "cons_loss": cons_loss,
                    }
                )
                eval_results[test_set].append(
                    {
                        "model": model,
                        "prefix": prefix,
                        "method": maybe_append_project_suffix(
                            method, project_along_mean_diff
                        ),
                        "prompt_level": prompt_idx,
                        "train": train_sets_str,
                        "test": test_set,
                        "location": _config["location"],
                        "layer": _config["layer"],
                        "accuracy": res[test_set][prompt_idx],
                        "ece": ece_dict[test_set][prompt_idx],
                        "ece_flip": ece_flip_dict[test_set][prompt_idx],
                    }
                )

    if _config["save_results"]:
        train_results_path = load_utils.get_train_results_path(run_dir)
        train_results_df = pd.DataFrame(train_results)
        train_results_df.to_csv(train_results_path, index=False)

        for ds, ds_eval_results in eval_results.items():
            eval_dir = load_utils.get_eval_dir(run_dir, ds, seed, run_id)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            eval_results_path = load_utils.get_eval_results_path(
                run_dir, ds, seed, run_id
            )
            ds_eval_results_df = pd.DataFrame(ds_eval_results)
            ds_eval_results_df.to_csv(eval_results_path, index=False)
