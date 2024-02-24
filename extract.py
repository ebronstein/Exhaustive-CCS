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

from utils import file_utils
from utils_extraction import load_utils
from utils_extraction.load_utils import (
    get_params_dir,
    load_hidden_states_for_datasets,
    make_permutation_dict,
    maybe_append_project_suffix,
    save_params,
)
from utils_extraction.method_utils import eval, mainResults
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


def get_method_str(method: str) -> str:
    return "CCS" if method.startswith("RCCS") else method


def get_project_along_mean_diff(
    method: str, project_along_mean_diff: bool
) -> bool:
    return "-md" in method or project_along_mean_diff


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

    # Evaluation

    # Dataset to evaluate the classification methods on.
    eval_datasets: Union[str, list[str]] = "imdb"
    # If true, load saved classifiers and only perform evaluation without
    # training. Otherwise, train and evaluate the classifiers from scratch.
    eval_only: bool = False

    load_params_save_dir: str = save_dir
    load_params_name: str = name
    # Run ID to load the parameters from. If classifiers are not being
    # loaded, this is None. If "latest", the latest run ID will be used.
    # Otherwise, the specified run ID will be used.
    load_params_run_id: Union[int, Literal["latest"]] = "latest"

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
    if config["prefix"] not in typing.get_args(PrefixType):
        raise ValueError(f"Invalid prefix: {config['prefix']}")
    if any([method.startswith("RCCS") for method in config["method_list"]]):
        raise NotImplementedError("RCCS is not yet implemented.")

    config = _convert_dogmatics_to_standard(config)

    # Convert single strings to lists.
    for key in ["datasets", "eval_datasets", "method_list"]:
        if isinstance(config[key], str):
            config[key] = [config[key]]

    config["location"] = parser_utils.get_states_location_str(
        config["location"], config["model"], use_auth_token=HF_AUTH_TOKEN
    )
    # TODO: validate location more extensively.
    if config["location"] == "decoder" and config["layer"] < 0:
        config["layer"] += config["num_layers"]

    return config


@ex.automain
def main(model, save_dir, exp_dir, _config: dict, seed: int, _log, _run):
    _config = _format_config(_config)
    train_datasets = _config["datasets"]
    eval_datasets = _config["eval_datasets"]
    prefix = _config["prefix"]

    run_id = _run._id
    run_dir = os.path.join(exp_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if _config["eval_only"]:
        load_exp_dir = load_utils.get_exp_dir(
            _config["load_params_save_dir"],
            _config["load_params_name"],
            model,
            train_datasets,
            seed,
        )
        load_params_run_id = _config["load_params_run_id"]
        if load_params_run_id == "latest":
            load_params_run_id = load_utils.maximum_existing_run_id(
                load_exp_dir
            )
        load_run_dir = os.path.join(load_exp_dir, str(load_params_run_id))

        # Check that the parameter directory exists for each method.
        for method in _config["method_list"]:
            method_str = get_method_str(method)
            project_along_mean_diff = get_project_along_mean_diff(
                method, _config["project_along_mean_diff"]
            )
            load_params_dir = get_params_dir(
                load_run_dir,
                maybe_append_project_suffix(
                    method_str, project_along_mean_diff
                ),
                prefix,
            )
            if not os.path.exists(load_params_dir):
                raise FileNotFoundError(
                    f"Directory {load_params_dir} does not exist. Cannot load parameters."
                )

    model_short_name = file_utils.get_model_short_name(model)
    _log.info(
        "---------------- model = %s, prefix = %s ----------------"
        % (model_short_name, prefix)
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
        #     dataset_list=train_datasets,
        #     prefix=prefix,
        # )
        # for setname in train_datasets:
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

    eval_results = collections.defaultdict(list)

    for method in _config["method_list"]:
        if method == "0-shot":
            continue
        _log.info("-------- method = %s --------", method)

        project_along_mean_diff = get_project_along_mean_diff(
            method, _config["project_along_mean_diff"]
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

        # If only evaluating, load just the eval datasets. Otherwise, load both
        # the train and eval datasets.
        if _config["eval_only"]:
            datasets_to_load = eval_datasets
        else:
            datasets_to_load = list(set(train_datasets + eval_datasets))
        data_dict = load_hidden_states_for_datasets(
            _config["load_dir"],
            mdl_name=model,
            dataset_list=datasets_to_load,
            prefix=prefix,
            location=_config["location"],
            layer=_config["layer"],
            mode=mode,
            logger=_log,
        )
        permutation_dict = {
            ds: make_permutation_dict(data_dict[ds]) for ds in datasets_to_load
        }
        assert data_dict.keys() == set(datasets_to_load)
        assert permutation_dict.keys() == set(datasets_to_load)

        # TODO: maybe projection should get to use other datasets that are
        # not in the train and eval list. projection_dict is currently being
        # used to specify the training datasets, so these two uses should be
        # separated.
        projection_datasets = (
            eval_datasets if _config["eval_only"] else train_datasets
        )
        projection_dict = {
            key: list(range(len(data_dict[key]))) for key in projection_datasets
        }

        test_dict = {ds: range(len(data_dict[ds])) for ds in eval_datasets}

        n_components = 1 if method == "TPC" else -1

        method_str = get_method_str(method)
        method_ = method
        constraints = None
        if method.startswith("RCCS"):
            method_ = "CCS"
            if method != "RCCS0":
                # TODO: load constraints and old_biases from the correct directory for
                # RCCS.
                prev_rccs_params_dir = None
                # TODO: replace this with call to parameter-loading function
                # and use a different directory where the previous CCS params
                # are stored.
                constraints = np.load(
                    os.path.join(prev_rccs_params_dir, "coef.npy")
                )
                old_biases = np.load(
                    os.path.join(prev_rccs_params_dir, "intercept.npy")
                )

        kwargs = dict(
            data_dict=data_dict,
            permutation_dict=permutation_dict,
            test_dict=test_dict,
            projection_dict=projection_dict,
            projection_method="PCA",
            n_components=n_components,
            prefix=prefix,
            classification_method=method_,
            print_more=_config["verbose"],
            save_probs=_config["save_states"],
            test_on_train=_config["test_on_train"],
            project_along_mean_diff=project_along_mean_diff,
            seed=seed,
        )
        if _config["eval_only"]:
            load_params_dir = get_params_dir(
                load_run_dir,
                maybe_append_project_suffix(
                    method_str, project_along_mean_diff
                ),
                prefix,
            )
            eval_kwargs = dict(
                run_dir=load_run_dir,
                run_id=_config["load_params_run_id"],
                **kwargs,
            )
            res, lss, ece, pmodel, cmodel = eval(**eval_kwargs)
        else:
            main_results_kwargs = dict(
                constraints=constraints,
                run_dir=run_dir,
                run_id=run_id,
                **kwargs,
            )
            res, lss, ece, pmodel, cmodel = mainResults(**main_results_kwargs)

        # Save parameters if needed.
        if (
            _config["save_params"]
            and not _config["eval_only"]
            and (
                method in ["TPC", "BSS", "CCS", "Random", "LR"]
                or method.startswith("RCCS")
            )
        ):
            save_params_dir = get_params_dir(
                run_dir,
                maybe_append_project_suffix(
                    method_str, project_along_mean_diff
                ),
                prefix,
            )
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
            elif method.startswith("RCCS"):
                coef_and_bias = cmodel.best_theta
                coef = coef_and_bias[:, :-1]
                bias = coef_and_bias[:, -1]
                if method != "RCCS0":
                    coef = np.concatenate([constraints, coef], axis=0)
                    bias = np.concatenate([old_biases, bias], axis=0)
            else:
                assert False

            save_params(
                save_params_dir,
                coef,
                bias,
            )

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

        train_sets_str = load_utils.get_combined_datasets_str(train_datasets)
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
        for test_set in eval_datasets:
            # TODO: handle the case where consecutive prompts may not be used,
            # so `res` should store the actual prompt indices rather than just
            # a list of the accuracy (same for the other results from
            # `mainResults`).
            for prompt_idx in range(len(res[test_set])):
                eval_results[test_set].append(
                    {
                        "model": model_short_name,
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
                        "loss": loss,
                        "sim_loss": sim_loss,
                        "cons_loss": cons_loss,
                    }
                )

    if _config["save_results"]:
        for ds, ds_eval_results in eval_results.items():
            eval_dir = load_utils.get_eval_dir(run_dir, ds)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            eval_results_path = load_utils.get_eval_results_path(run_dir, ds)
            ds_eval_results_df = pd.DataFrame(ds_eval_results)
            ds_eval_results_df.to_csv(eval_results_path, index=False)
