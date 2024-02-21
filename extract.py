import argparse
import collections
import json
import os
import random
import time
import warnings
from typing import Any, Iterable, Literal, Union

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.config.custom_containers import DogmaticDict, DogmaticList
from sacred.observers import FileStorageObserver

from utils.file_utils import get_model_short_name
from utils_extraction import load_utils
from utils_extraction.func_utils import eval_adder, getAvg, train_adder
from utils_extraction.load_utils import (
    get_params_dir,
    get_probs_save_path,
    get_results_save_path,
    get_zeros_acc,
    getDic,
)
from utils_extraction.method_utils import is_method_unsupervised, mainResults
from utils_generation import hf_utils
from utils_generation import parser as parser_utils
from utils_generation.hf_auth_token import HF_AUTH_TOKEN
from utils_generation.save_utils import maybeAppendProjectSuffix, saveParams

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

ex = Experiment()


def methodHasLoss(method):
    return method in ["BSS", "CCS"] or method.startswith("RCCS")


@ex.config
def sacred_config():
    # Experiment name
    name: str = "extraction"
    model = "/scratch/data/meta-llama/Llama-2-7b-chat-hf"
    datasets: Union[str, list[str]] = "imdb"
    prefix: PrefixType = "normal"
    data_num: int = 1000
    method_list: Literal["0-shot", "TPC", "KMeans", "LR", "BSS", "CCS"] = "CCS"
    mode: Literal["auto", "minus", "concat"] = "auto"
    save_dir = "extraction_results"
    load_dir = "generation_results"
    load_classifier: bool = False
    params_load_dir: str = "extraction_results"
    location: Literal["auto", "encoder", "decoder"] = "auto"
    layer: int = -1
    num_layers = hf_utils.get_num_hidden_layers(model)
    # File name where zero-shot results will be saved.
    zero: str = "zero_shot"
    seed: int = 0
    prompt_save_level: Literal["single", "all"] = "all"
    save_states: bool = True
    save_params = True
    # TODO: change this to save_results
    no_save_results: bool = False
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

    run_dir = os.path.join(exp_dir, _run._id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if _config["save_params"]:
        load_utils.make_params_dir(run_dir)

    prefix = _config["prefix"]
    _log.info(
        "---------------- model = {}, prefix = {} ----------------".format(
            model, prefix
        )
    )

    # Start calculate numbers
    # std is over all prompts within this dataset
    train_results_path = load_utils.get_train_results_path(run_dir)
    eval_results_path = load_utils.get_eval_results_path(run_dir)

    train_csv = pd.DataFrame(
        columns=[
            "model",
            "prefix",
            "method",
            "prompt_level",
            "train",
            "test",
        ]
    )
    eval_csv = pd.DataFrame(
        columns=[
            "model",
            "prefix",
            "method",
            "prompt_level",
            "train",
            "test",
            "accuracy",
            "std",
        ]
    )

    # TODO: look into how zero-shot results are being saved.
    if "0-shot" in _config["method_list"]:
        raise NotImplementedError(
            "Zero-shot extraction is not yet implemented."
        )
        # load zero-shot performance
        rawzeros = pd.read_csv(
            os.path.join(_config["load_dir"], "{}.csv".format(_config["zero"]))
        )
        # Get the global zero acc dict (setname, [acc])
        zeros_acc = get_zeros_acc(
            _config["load_dir"],
            csv_name=_config["zero"],
            mdl_name=model,
            dataset_list=dataset_list,
            prefix=prefix,
        )
        for setname in dataset_list:
            if _config["prompt_save_level"] == "all":
                eval_csv = eval_adder(
                    eval_csv,
                    model,
                    prefix,
                    "0-shot",
                    "",
                    "",
                    setname,
                    np.mean(zeros_acc[setname]),
                    np.std(zeros_acc[setname]),
                    "",
                    "",
                    "",
                    "",
                )
            else:  # For each prompt, save one line
                for idx in range(len(zeros_acc[setname])):
                    eval_csv = eval_adder(
                        eval_csv,
                        model,
                        prefix,
                        "0-shot",
                        prompt_level=idx,
                        train="",
                        test=setname,
                        accuracy=zeros_acc[setname][idx],
                        std="",
                        ece="",
                        location="",
                        layer="",
                        loss="",
                        sim_loss="",
                        cons_loss="",
                    )

        if not _config["no_save_results"]:
            eval_csv.to_csv(eval_results_path, index=False)
            if _config["verbose"]:
                _log.info(
                    "Saved zero-shot performance to %s", eval_results_path
                )

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
        data_dict, permutation_dict = getDic(
            _config["load_dir"],
            mdl_name=model,
            dataset_list=dataset_list,
            prefix=prefix,
            location=_config["location"],
            layer=_config["layer"],
            mode=mode,
        )
        assert data_dict.keys() == set(dataset_list)
        assert permutation_dict.keys() == set(dataset_list)

        test_dict = {ds: range(len(data_dict[ds])) for ds in dataset_list}

        train_set = "all"
        train_list = dataset_list
        projection_dict = {
            key: range(len(data_dict[key])) for key in train_list
        }

        n_components = 1 if method == "TPC" else -1

        method_str = "CCS" if method.startswith("RCCS") else method
        params_dir = get_params_dir(
            run_dir,
            maybeAppendProjectSuffix(method_str, project_along_mean_diff),
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

        load_classifier_dir_and_name = (
            (_config["params_load_dir"], params_dir)
            if _config["load_classifier"]
            else None
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
            load_classifier_dir_and_name=load_classifier_dir_and_name,
            classification_method=method_,
            print_more=_config["verbose"],
            save_probs=_config["save_states"],
            test_on_train=_config["test_on_train"],
            constraints=constraints,
            project_along_mean_diff=project_along_mean_diff,
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
                saveParams(
                    _config["save_dir"],
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
                saveParams(_config["save_dir"], params_dir, coef, bias)

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
                maybeAppendProjectSuffix(method, project_along_mean_diff),
                "all",
                train_set,
                100 * acc,
                100 * std,
                loss,
                sim_loss,
                cons_loss,
                mean_ece,
                mean_ece_flip,
            )
        )

        for key in dataset_list:
            if _config["prompt_save_level"] == "all":
                loss, sim_loss, cons_loss = (
                    np.mean(lss[key], axis=0)
                    if methodHasLoss(method)
                    else ("", "", "")
                )
                train_csv = train_adder(
                    train_csv,
                    model,
                    prefix,
                    maybeAppendProjectSuffix(method, project_along_mean_diff),
                    "all",
                    train_set,
                    key,
                    location=_config["location"],
                    layer=_config["layer"],
                    loss=loss,
                    sim_loss=sim_loss,
                    cons_loss=cons_loss,
                )
                eval_csv = eval_adder(
                    eval_csv,
                    model,
                    prefix,
                    maybeAppendProjectSuffix(method, project_along_mean_diff),
                    "all",
                    train_set,
                    key,
                    accuracy=np.mean(res[key]),
                    std=np.std(res[key]),
                    ece=np.mean(ece_dict[key]),
                    ece_flip=np.mean(ece_flip_dict[key]),
                    location=_config["location"],
                    layer=_config["layer"],
                )
            else:
                for idx in range(len(res[key])):
                    loss, sim_loss, cons_loss = (
                        lss[key][idx] if methodHasLoss(method) else ("", "", "")
                    )
                    train_csv = train_adder(
                        train_csv,
                        model,
                        prefix,
                        maybeAppendProjectSuffix(
                            method, project_along_mean_diff
                        ),
                        idx,
                        train_set,
                        key,
                        layer=_config["layer"],
                        loss=loss,
                        sim_loss=sim_loss,
                        cons_loss=cons_loss,
                    )
                    eval_csv = eval_adder(
                        eval_csv,
                        model,
                        prefix,
                        maybeAppendProjectSuffix(
                            method, project_along_mean_diff
                        ),
                        idx,
                        train_set,
                        key,
                        accuracy=res[key][idx],
                        std="",
                        ece=ece_dict[key][idx],
                        ece_flip=ece_flip_dict[key][idx],
                        location=_config["location"],
                        layer=_config["layer"],
                    )

    if not _config["no_save_results"]:
        train_csv.to_csv(train_results_path, index=False)
        eval_csv.to_csv(eval_results_path, index=False)
