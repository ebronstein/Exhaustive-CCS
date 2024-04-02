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
from utils_extraction.classifier import SpanDirsCombination
from utils_extraction.load_utils import (
    get_params_dir,
    load_hidden_states_for_datasets,
    make_permutation_dict,
    maybe_append_project_suffix,
    replace_burns_datasets,
    save_fit_plots,
    save_fit_result,
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
    "normal-bananashed",
]

MethodType = Literal[
    "0-shot",
    "TPC",
    "KMeans",
    "LR",
    "BSS",
    "CCS",
    "CCS+LR",
    "CCS-in-LR-span",
    "CCS+LR-in-span",
    "CCS-select-LR",
]

ex = Experiment()


def get_method_str(method: str) -> str:
    return "CCS" if method.startswith("RCCS") else method


def get_project_along_mean_diff(method: str, project_along_mean_diff: bool) -> bool:
    return "-md" in method or project_along_mean_diff


def method_uses_concat_hs_mode(method: str) -> bool:
    return (
        method in ("LR", "CCS", "CCS+LR", "CCS-in-LR-span", "CCS+LR-in-span", "CCS-select-LR", "Random")
    ) or method.startswith("RCCS")


def default_method_mode(method: str) -> str:
    return "concat" if method_uses_concat_hs_mode(method) else "minus"


def getAvg(dic):
    return np.mean([np.mean(lis) for lis in dic.values()])


@ex.config
def sacred_config():
    # Experiment name
    name: str = "extraction"
    model = "/scratch/data/meta-llama/Llama-2-7b-chat-hf"
    # Main training datasets. Set to "burns" to use all the Burns datasets.
    datasets: Union[str, list[str]] = "imdb"
    # Optional second set of labeled training datasets. If provided, `datasets`
    # is used for unsupervised learning and `labeled_datasets` is used for
    # supervised learning. The method must support the combination of the two.
    # Set to "burns" to use all the Burns datasets.
    labeled_datasets: Union[str, list[str]] = []
    # Prefix to use for training.
    prefix: PrefixType = "normal"
    # Prefix to use for evaluation. If None, the training prefix will be used.
    test_prefix: Optional[PrefixType] = None
    data_num: int = 1000
    mode: Literal["auto", "minus", "concat"] = "auto"
    load_dir = "generation_results"
    location: Literal["auto", "encoder", "decoder"] = "auto"
    layer: int = -1
    num_layers = hf_utils.get_num_hidden_layers(model, use_auth_token=HF_AUTH_TOKEN)
    # File name where zero-shot results will be saved.
    zero: str = "zero_shot"
    seed: int = 0
    project_along_mean_diff: bool = False
    verbose: bool = False

    # Training
    method_list: Union[MethodType, list[MethodType]] = "CCS"
    n_tries: int = 10
    n_epochs: int = 1000
    sup_weight: float = 1.0
    unsup_weight: float = 1.0
    consistency_weight: float = 1.0
    confidence_weight: float = 1.0
    lr: float = 1e-2
    include_bias: bool = True
    opt: Literal["sgd", "adam"] = "sgd"
    num_orthogonal_directions: int = 4
    # Run directory or datasets ancestor directory to load orthogonal
    # directions from. If provided, can be the run directory, in which
    # case the orthogonal directions should be at
    # {load_orthogonal_directions_dir}/train/orthogonal_directions.npy, or the
    # datasets ancestory directory, in which case the orthogonal directions
    # should be at
    # {load_orthogonal_directions_dir}/seed_{seed}/{run_dir}/train/orthogonal_directions.npy.
    # For the latter, if there are multiple runs, the latest run will be used.
    # If None, orthogonal directions will be trained from scratch.
    load_orthogonal_directions_dir: Optional[str] = None
    span_dirs_combination: SpanDirsCombination = "linear"
    device: Literal["cuda", "cpu"] = "cuda"
    # Logistic regression parameters. See sklearn.linear_model.LogisticRegression.
    log_reg = {
        "penalty": "l2",
        "C": 0.1,
        "max_iter": 10_000,
        "fit_intercept": True,
    }

    # Saving
    save_dir = "extraction_results"
    save_fit_result: bool = True
    save_fit_plots: bool = False
    save_states: bool = True
    save_params = True
    save_results: bool = True
    save_train_test_split: bool = True
    save_orthogonal_directions: bool = False

    # Evaluation

    # Dataset to evaluate the classification methods on. Set to "burns" to use
    # all the Burns datasets.
    eval_datasets: Union[str, list[str]] = "imdb"
    # If true, load saved classifiers and only perform evaluation without
    # training. Otherwise, train and evaluate the classifiers from scratch.
    eval_only: bool = False
    test_on_train: bool = False

    load_params_save_dir: str = save_dir
    load_params_name: str = name
    # Run ID to load the parameters from. If classifiers are not being
    # loaded, this is None. If "latest", the latest run ID will be used.
    # Otherwise, the specified run ID will be used.
    load_params_run_id: Union[int, Literal["latest"]] = "latest"

    exp_dir = load_utils.get_exp_dir(
        save_dir, name, model, datasets, seed, labeled_datasets=labeled_datasets
    )
    # NOTE: set the observers to this single FileStorageObserver instead of
    # appending to ex.observers. This allows running the experiment multiple
    # times without creating multiple observers.
    ex.observers = [FileStorageObserver(exp_dir, copy_sources=False)]


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

    # Convert single strings to lists and remove duplicates.
    for key in ["datasets", "labeled_datasets", "eval_datasets", "method_list"]:
        if isinstance(config[key], str):
            config[key] = [config[key]]
        else:
            config[key] = list(set(config[key]))

    # Replace Burns datasets.
    for key in ["datasets", "labeled_datasets", "eval_datasets"]:
        config[key] = replace_burns_datasets(config[key])

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
    labeled_train_datasets = _config["labeled_datasets"]
    eval_datasets = _config["eval_datasets"]
    prefix = _config["prefix"]
    test_prefix = (
        _config["test_prefix"] if _config["test_prefix"] is not None else prefix
    )

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
            labeled_datasets=labeled_train_datasets,
        )
        load_params_run_id = _config["load_params_run_id"]
        if load_params_run_id == "latest":
            load_params_run_id = load_utils.maximum_existing_run_id(load_exp_dir)
        load_run_dir = os.path.join(load_exp_dir, str(load_params_run_id))

        # Check that the parameter directory exists for each method.
        for method in _config["method_list"]:
            method_str = get_method_str(method)
            project_along_mean_diff = get_project_along_mean_diff(
                method, _config["project_along_mean_diff"]
            )
            full_method_str = maybe_append_project_suffix(
                method_str, project_along_mean_diff
            )
            load_params_dir = get_params_dir(
                load_run_dir,
                full_method_str,
                prefix,
            )
            if not os.path.exists(load_params_dir):
                raise FileNotFoundError(
                    f"Directory {load_params_dir} does not exist. Cannot load parameters."
                )

    model_short_name = file_utils.get_model_short_name(model)
    _log.info(
        "---------------- model = %s, prefix = %s, test_prefix - %s ----------------"
        % (model_short_name, prefix, test_prefix)
    )

    # TODO: look into how zero-shot results are being saved.
    if "0-shot" in _config["method_list"]:
        raise NotImplementedError("Zero-shot extraction is not yet implemented.")
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

    # If only evaluating, load just the eval datasets. Otherwise, load both
    # the train and eval datasets.
    if _config["eval_only"]:
        datasets_to_load = eval_datasets
    else:
        datasets_to_load = list(
            set(train_datasets + labeled_train_datasets + eval_datasets)
        )

    data_dict = None
    eval_results = collections.defaultdict(list)

    # Generate data.
    prefixes = set([prefix, test_prefix])
    mode_to_data = {}
    for method in _config["method_list"]:
        if method == "0-shot":
            continue

        project_along_mean_diff = get_project_along_mean_diff(
            method, _config["project_along_mean_diff"]
        )
        if project_along_mean_diff:
            method = method.replace("-md", "")

        mode = (
            _config["mode"]
            if _config["mode"] != "auto"
            else default_method_mode(method)
        )
        if mode in mode_to_data:
            continue

        mode_to_data[mode] = {}
        for prefix_ in prefixes:
            # Only generate the data (and other related dictionaries) once to keep
            # them the same across methods.
            data_dict = load_hidden_states_for_datasets(
                _config["load_dir"],
                mdl_name=model,
                dataset_list=datasets_to_load,
                prefix=prefix_,
                location=_config["location"],
                layer=_config["layer"],
                mode=mode,
                logger=_log,
            )
            mode_to_data[mode][prefix_] = data_dict

    for method in _config["method_list"]:
        if method == "0-shot":
            continue
        _log.info("-------- method = %s --------", method)

        project_along_mean_diff = get_project_along_mean_diff(
            method, _config["project_along_mean_diff"]
        )
        if project_along_mean_diff:
            method = method.replace("-md", "")

        mode = (
            _config["mode"]
            if _config["mode"] != "auto"
            else default_method_mode(method)
        )

        data_dict = mode_to_data[mode]
        permutation_dict = {
            ds: make_permutation_dict(data_dict[prefix][ds]) for ds in datasets_to_load
        }
        assert data_dict[prefix].keys() == set(datasets_to_load)
        assert permutation_dict.keys() == set(datasets_to_load)

        # TODO: maybe projection should get to use other datasets that are
        # not in the train and eval list. projection_dict is currently being
        # used to specify the training datasets, so these two uses should be
        # separated.
        projection_datasets = (
            eval_datasets
            if _config["eval_only"]
            else set(train_datasets + labeled_train_datasets)
        )
        # Arbitrarily use prefix instead of test_prefix to index into data_dict
        # since the number of prompts should be the same for both.
        projection_dict = {
            ds: list(range(len(data_dict[prefix][ds]))) for ds in projection_datasets
        }

        train_data_dict = {
            ds: range(len(data_dict[prefix][ds])) for ds in train_datasets
        }
        labeled_train_data_dict = {
            ds: range(len(data_dict[prefix][ds])) for ds in labeled_train_datasets
        }

        test_dict = {ds: range(len(data_dict[test_prefix][ds])) for ds in eval_datasets}

        n_components = 1 if method == "TPC" else -1

        method_str = get_method_str(method)
        full_method_str = maybe_append_project_suffix(
            method_str, project_along_mean_diff
        )
        method_ = method
        if method.startswith("RCCS"):
            method_ = "CCS"
            if method != "RCCS0":
                # TODO: load constraints and old_biases from the correct directory for
                # RCCS.
                prev_rccs_params_dir = None
                # TODO: replace this with call to parameter-loading function
                # and use a different directory where the previous CCS params
                # are stored.
                constraints = np.load(os.path.join(prev_rccs_params_dir, "coef.npy"))
                old_biases = np.load(
                    os.path.join(prev_rccs_params_dir, "intercept.npy")
                )
        else:
            constraints = None

        train_kwargs = dict(
            n_tries=_config["n_tries"],
            n_epochs=_config["n_epochs"],
            sup_weight=_config["sup_weight"],
            unsup_weight=_config["unsup_weight"],
            consistency_weight=_config["consistency_weight"],
            confidence_weight=_config["confidence_weight"],
            lr=_config["lr"],
            include_bias=_config["include_bias"],
            opt=_config["opt"],
            num_orthogonal_directions=_config["num_orthogonal_directions"],
            span_dirs_combination=_config["span_dirs_combination"],
            log_reg=_config["log_reg"],
        )
        kwargs = dict(
            train_data_dict=train_data_dict,
            permutation_dict=permutation_dict,
            test_dict=test_dict,
            projection_dict=projection_dict,
            mode=mode,
            labeled_train_data_dict=labeled_train_data_dict,
            projection_method="PCA",
            n_components=n_components,
            classification_method=method_,
            train_kwargs=train_kwargs,
            print_more=_config["verbose"],
            save_probs=_config["save_states"],
            test_on_train=_config["test_on_train"],
            project_along_mean_diff=project_along_mean_diff,
            seed=seed,
            device=_config["device"],
            logger=_log,
        )
        if _config["eval_only"]:
            load_params_dir = get_params_dir(
                load_run_dir,
                full_method_str,
                prefix,
            )
            eval_kwargs = dict(
                data_dict=data_dict[test_prefix],
                train_prefix=prefix,
                run_dir=load_run_dir,
                run_id=_config["load_params_run_id"],
                **kwargs,
            )
            acc_dict, loss_dict, ece_dict, proj_model, classifier = eval(**eval_kwargs)
        else:
            if _config["load_orthogonal_directions_dir"] is not None:
                load_orthogonal_directions_run_dir = (
                    load_utils.get_orthogonal_directions_run_dir(
                        _config["load_orthogonal_directions_dir"], seed, logger=_log
                    )
                )
            else:
                load_orthogonal_directions_run_dir = None
            main_results_kwargs = dict(
                data_dict=data_dict,
                train_prefix=prefix,
                test_prefix=test_prefix,
                constraints=constraints,
                run_dir=run_dir,
                run_id=run_id,
                save_orthogonal_directions=_config["save_orthogonal_directions"],
                load_orthogonal_directions_run_dir=load_orthogonal_directions_run_dir,
                **kwargs,
            )
            (
                acc_dict,
                loss_dict,
                ece_dict,
                proj_model,
                classifier,
                fit_result,
            ) = mainResults(**main_results_kwargs)

        if fit_result:
            if _config["save_fit_result"]:
                fit_result = {
                    k: v
                    for k, v in fit_result.items()
                    if k not in ("best_probe", "all_probes")
                }
                save_fit_result(fit_result, run_dir, method, logger=_log)
            if _config["save_fit_plots"]:
                save_fit_plots(fit_result, run_dir, method, logger=_log)

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
                maybe_append_project_suffix(method_str, project_along_mean_diff),
                prefix,
            )
            if method in ["TPC", "BSS"]:
                coef, bias = (
                    classifier.coef_ @ proj_model.getDirection(),
                    classifier.intercept_,
                )
            elif method in ["CCS", "Random"]:
                coef = classifier.coef
                bias = classifier.bias
            elif method == "LR":
                coef, bias = classifier.coef_, classifier.intercept_
            elif method.startswith("RCCS"):
                coef_and_bias = classifier.best_theta
                coef = coef_and_bias[:, :-1]
                bias = coef_and_bias[:, -1]
                if method != "RCCS0":
                    coef = np.concatenate([constraints, coef], axis=0)
                    bias = np.concatenate([old_biases, bias], axis=0)
            # TODO: save CSS+LR using torch.save.
            elif method in ["CSS+LR", "CCS-in-LR-span", "CCS+LR-in-span", "CCS-select-LR"]:
                raise NotImplementedError()
            else:
                raise ValueError(f"Invalid method: {method}")

            save_params(
                save_params_dir,
                coef,
                bias,
            )

        if _config["save_train_test_split"]:
            load_utils.save_permutation_dict(permutation_dict, run_dir)

        acc = getAvg(acc_dict)
        std = np.mean([np.std(lis) for lis in acc_dict.values()])

        # TODO: standardize losses
        # Mean losses over all eval datasets and all prompts.
        if method in ["CCS+LR", "CCS-in-LR-span", "CCS+LR-in-span", "CCS-select-LR"]:
            loss_names = list(loss_dict[list(loss_dict.keys())[0]][0].keys())
            mean_losses = {}
            for loss_name in loss_names:
                mean_losses[loss_name] = np.mean(
                    [
                        np.mean([losses[loss_name] for losses in test_set_losses])
                        for test_set_losses in loss_dict.values()
                    ]
                )
        elif loss_dict:
            mean_losses = np.mean(
                [np.mean(lis, axis=0) for lis in loss_dict.values()], axis=0
            )

        og_ece_dict = ece_dict
        ece_dict = {
            key: [ece[0] for ece in ece_vals] for key, ece_vals in og_ece_dict.items()
        }
        ece_flip_dict = {
            key: [ece[1] for ece in ece_vals] for key, ece_vals in og_ece_dict.items()
        }
        mean_ece = getAvg(ece_dict)
        mean_ece_flip = getAvg(ece_flip_dict)

        train_sets_str = load_utils.get_combined_datasets_str(
            train_datasets, labeled_train_datasets
        )
        _log.info(
            "method = {:8}, prompt_level = {:8}, train_set = {:10}, avgacc is {:.2f}, std is {:.2f}, mean_losses are {}, ECE is {:.4f}, ECE (1-p) is {:.4f}".format(
                maybe_append_project_suffix(method, project_along_mean_diff),
                "all",
                train_sets_str,
                100 * acc,
                100 * std,
                mean_losses,
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
            for prompt_idx in range(len(acc_dict[test_set])):
                eval_result = {
                    "model": model_short_name,
                    "train_prefix": prefix,
                    "test_prefix": test_prefix,
                    "method": maybe_append_project_suffix(
                        method, project_along_mean_diff
                    ),
                    "prompt_level": prompt_idx,
                    "mode": mode,
                    "train": train_sets_str,
                    "test": test_set,
                    "location": _config["location"],
                    "layer": _config["layer"],
                    "accuracy": acc_dict[test_set][prompt_idx],
                    "ece": ece_dict[test_set][prompt_idx],
                    "ece_flip": ece_flip_dict[test_set][prompt_idx],
                }
                # TODO: standardize losses.
                if not loss_dict:
                    continue

                if method in ["CCS+LR", "CCS-in-LR-span", "CCS+LR-in-span", "CCS-select-LR"]:
                    for loss_name, loss in loss_dict[test_set][prompt_idx].items():
                        eval_result[loss_name] = loss
                elif "CCS" in method:
                    loss, sim_loss, cons_loss = loss_dict[test_set][prompt_idx]
                    eval_result["loss"] = loss
                    eval_result["sim_loss"] = sim_loss
                    eval_result["cons_loss"] = cons_loss
                eval_results[test_set].append(eval_result)

    if _config["save_results"]:
        for ds, ds_eval_results in eval_results.items():
            eval_dir = load_utils.get_eval_dir(run_dir, ds)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

            # Save the evaluation results to a CSV file.
            eval_results_path = load_utils.get_eval_results_path(run_dir, ds)
            ds_eval_results_df = pd.DataFrame(ds_eval_results)
            ds_eval_results_df.to_csv(eval_results_path, index=False)
