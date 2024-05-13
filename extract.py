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
from utils_extraction.classifier import ContrastPairClassifier, SpanDirsCombination
from utils_extraction.load_utils import (
    ALL_DATASETS,
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
from utils_extraction.pseudo_label import validate_pseudo_label_config
from utils_generation import hf_utils
from utils_generation import parser as parser_utils
from utils_generation.construct_prompts import (
    PROMPT_SUBSETS,
    get_global_prompts_num,
    prompt_name_to_index,
)
from utils_generation.hf_auth_token import HF_AUTH_TOKEN

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
    "bananashed",
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
    "pseudolabel",
]

ex = Experiment()


def get_method_str(method: str) -> str:
    return "CCS" if method.startswith("RCCS") else method


def get_project_along_mean_diff(method: str, project_along_mean_diff: bool) -> bool:
    return "-md" in method or project_along_mean_diff


def method_uses_concat_hs_mode(method: str) -> bool:
    return (
        method
        in (
            "LR",
            "CCS",
            "CCS+LR",
            "CCS-in-LR-span",
            "CCS+LR-in-span",
            "CCS-select-LR",
            "Random",
        )
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
    # Prefix for main training data (`datasets`).
    prefix: PrefixType = "normal"
    # Prefix for labeled training data (`labeled_datasets`). If not provided and
    # labeled datasets are used, defaults to `prefix`.
    labeled_prefix: Optional[PrefixType] = None
    # Prefix to use for evaluation. If None, the training prefix will be used.
    test_prefix: Optional[PrefixType] = None
    # Prompt indices or names for the main training data. prompt_idx[dataset] is
    # a list of prompt indices or names for `dataset`. If an element is an int,
    # it is interpreted as an index, otherwise as a name If None, all default
    # prompts will be used.
    prompt_idx: Optional[dict[str, list[int]]] = None
    prompt_subset: Optional[str] = "default"
    # Prompt indices or names for the labeled training data. If None, all
    # default prompts will be used.
    labeled_prompt_idx: Optional[dict[str, list[int]]] = None
    labeled_prompt_subset: Optional[str] = "default"
    # Prompt indices or names for the evaluation data. If None, all prompts will
    # be used. Defaults to all prompts (not just default prompts).
    test_prompt_idx: Optional[dict[str, list[int]]] = None
    test_prompt_subset: Optional[str] = "all"
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
    weight_decay: float = 0.0
    opt: Literal["sgd", "adam"] = "sgd"
    num_orthogonal_directions: int = 4
    projected_sgd: bool = True
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
    span_dirs_combination: SpanDirsCombination = "convex"
    device: Literal["cuda", "cpu"] = "cuda"
    # Logistic regression parameters. See sklearn.linear_model.LogisticRegression.
    log_reg = {
        "penalty": "l2",
        "C": 0.1,
        "max_iter": 10_000,
        "fit_intercept": True,
    }
    # Pseudo-labeling method parameters.
    pseudolabel = dict(
        n_rounds=1,
        select_fn="high_confidence_consistency",
        prob_threshold=0.8,
        # Threshold for the consistency error.
        consistency_err_threshold=0.1,
        label_fn="argmax",
        # Temperature for the softmax function when creating pseudolabels
        # for label_fn="softmax".
        softmax_temp=0.3,
    )

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


def _validate_config(config: dict) -> None:
    for key in ["prefix", "labeled_prefix"]:
        if config[key] not in typing.get_args(PrefixType):
            raise ValueError(f"Invalid {key}: {config[key]}")

    if any([method.startswith("RCCS") for method in config["method_list"]]):
        raise NotImplementedError("RCCS is not yet implemented.")

    validate_pseudo_label_config(config["pseudolabel"])


def _format_config(config: dict) -> dict:
    config = _convert_dogmatics_to_standard(config)

    # Set default prefixes to the main training data prefix.
    for key in ["labeled_prefix", "test_prefix"]:
        if config[key] is None:
            config[key] = config["prefix"]

    # Convert single strings to lists and remove duplicates.
    for key in ["datasets", "labeled_datasets", "eval_datasets", "method_list"]:
        ds_list = config[key]
        if isinstance(ds_list, str):
            ds_list = [ds_list]
        config[key] = list(set(ds_list))

    # Replace Burns datasets.
    for key in ["datasets", "labeled_datasets", "eval_datasets"]:
        config[key] = list(set(replace_burns_datasets(config[key])))

    # Set prompt indices based on prompt subsets if provided.
    for idx_key, subset_key, dataset_key in zip(
        ["prompt_idx", "labeled_prompt_idx", "test_prompt_idx"],
        ["prompt_subset", "labeled_prompt_subset", "test_prompt_subset"],
        ["datasets", "labeled_datasets", "eval_datasets"],
    ):
        prompt_subset = config[subset_key]
        if prompt_subset is not None:
            prompt_idx = config[idx_key]
            assert (
                prompt_idx is None
            ), f"Cannot specify both {subset_key} and {idx_key}."
            # Set the prompt indices to the subset.
            config[idx_key] = {
                ds: PROMPT_SUBSETS[ds][prompt_subset] for ds in config[dataset_key]
            }

    # Process prompt indices:
    # 1. Make sure they are not duplicated.
    # 2. Convert names to indices.
    for prompt_idx_dict in [
        config["prompt_idx"],
        config["labeled_prompt_idx"],
        config["test_prompt_idx"],
    ]:
        if prompt_idx_dict is None:
            continue

        for ds, prompt_idxs in prompt_idx_dict.items():
            if prompt_idxs is None:
                continue

            processed_prompt_idxs = []
            for idx_or_name in set(prompt_idxs):
                if isinstance(idx_or_name, int):
                    idx = idx_or_name
                else:
                    idx = prompt_name_to_index(idx_or_name, ds)
                    if idx is None:
                        raise ValueError(
                            f"Prompt name {idx_or_name} not found for dataset {ds}."
                        )
                processed_prompt_idxs.append(idx)

            prompt_idx_dict[ds] = sorted(processed_prompt_idxs)

    config["location"] = parser_utils.get_states_location_str(
        config["location"], config["model"], use_auth_token=HF_AUTH_TOKEN
    )
    # TODO: validate location more extensively.
    if config["location"] == "decoder" and config["layer"] < 0:
        config["layer"] += config["num_layers"]

    return config


def _check_config_subset_unmodified(config: dict, subset: dict) -> bool:
    """Check if the subset of the config is unmodified from the original config."""
    for key, value in subset.items():
        if key not in config or config[key] != value:
            return False
    return True


@ex.automain
def main(model, save_dir, exp_dir, _config: dict, seed: int, _log, _run):
    _config = _format_config(_config)
    _validate_config(_config)

    train_datasets = _config["datasets"]
    labeled_train_datasets = _config["labeled_datasets"]
    eval_datasets = _config["eval_datasets"]
    prefix = _config["prefix"]
    labeled_prefix = _config["labeled_prefix"]
    test_prefix = _config["test_prefix"]
    prompt_idx = _config["prompt_idx"]
    labeled_prompt_idx = _config["labeled_prompt_idx"]
    test_prompt_idx = _config["test_prompt_idx"]
    load_params_run_id = _config["load_params_run_id"]

    run_id = _run._id
    # If the run ID is not set, Sacred observers have been
    # intentionally removed.
    # TODO: handle this case in other places as well.
    if run_id is not None:
        run_dir = os.path.join(exp_dir, run_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if _config["eval_only"]:
        # TODO: adapt this handle using different prefix, labeled_prefix, and
        # test_prefix.
        load_exp_dir = load_utils.get_exp_dir(
            _config["load_params_save_dir"],
            _config["load_params_name"],
            model,
            train_datasets,
            seed,
            labeled_datasets=labeled_train_datasets,
        )
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
                labeled_prefix=labeled_prefix,
            )
            if not os.path.exists(load_params_dir):
                raise FileNotFoundError(
                    f"Directory {load_params_dir} does not exist. Cannot load parameters."
                )

    model_short_name = file_utils.get_model_short_name(model)
    _log.info(
        "---------------- model = %s, prefix = %s, labeled_prefix = %s, test_prefix - %s ----------------"
        % (model_short_name, prefix, labeled_prefix, test_prefix)
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

    prefixes = set([prefix, labeled_prefix, test_prefix])
    # Get the datasets that should be loaded and for which specific prompt
    # indices are requested. If the dataset should not be loaded, there is no
    # need to get prompts for it. Otherwise, if the dataset is not present in
    # the three prompt index dictionaries, the default behavior is to include
    # all the prompts for that dataset.
    datasets_for_prompt_idx = set(datasets_to_load)
    for ds_to_prompt_idx_dict in [prompt_idx, labeled_prompt_idx, test_prompt_idx]:
        if ds_to_prompt_idx_dict is not None:
            datasets_for_prompt_idx &= set(ds_to_prompt_idx_dict.keys())
    # Collect all prompt indices for each dataset.
    ds_to_prompt_idxs = collections.defaultdict(set)
    for ds_to_prompt_idx_dict in [prompt_idx, labeled_prompt_idx, test_prompt_idx]:
        if ds_to_prompt_idx_dict is None:
            continue
        for ds, prompt_idxs in ds_to_prompt_idx_dict.items():
            if ds in datasets_for_prompt_idx:
                ds_to_prompt_idxs[ds].update(prompt_idxs)
    # If any of the prompt indices are None, set the prompt indices for that
    # dataset to None to include all of its prompts.
    for ds, prompt_idxs in ds_to_prompt_idxs.items():
        if None in prompt_idxs:
            ds_to_prompt_idxs[ds] = None
    # Set all datasets without prompt indices to use all prompts.
    for ds in datasets_to_load:
        if ds not in ds_to_prompt_idxs:
            ds_to_prompt_idxs[ds] = None

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
                prompt_dict=ds_to_prompt_idxs,
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
            prefix_: {
                ds: make_permutation_dict(data_dict[prefix_][ds])
                for ds in datasets_to_load
            }
            for prefix_ in prefixes
        }
        for prefix_ in prefixes:
            assert data_dict[prefix_].keys() == set(datasets_to_load)
            assert permutation_dict[prefix_].keys() == set(datasets_to_load)

        # TODO: maybe projection should get to use other datasets that are
        # not in the train and eval list. projection_dict is currently being
        # used to specify the training datasets, so these two uses should be
        # separated.
        projection_datasets = (
            eval_datasets
            if _config["eval_only"]
            else set(train_datasets + labeled_train_datasets)
        )
        # TODO: use prompt_idx, labeled_prompt_idx, and test_prompt_idx somehow
        # for the projection datasets as well.
        # Arbitrarily use prefix instead of test_prefix to index into data_dict
        # since the number of prompts should be the same for both.
        projection_dict = {
            ds: list(range(len(data_dict[prefix][ds]))) for ds in projection_datasets
        }

        # Main training data prompt indices.
        train_data_dict = {}
        for ds in train_datasets:
            if prompt_idx is not None and prompt_idx.get(ds) is not None:
                train_data_dict[ds] = prompt_idx[ds]
            else:
                train_data_dict[ds] = range(len(data_dict[prefix][ds]))
        # Labeled training data prompt indices.
        labeled_train_data_dict = {}
        for ds in labeled_train_datasets:
            if (
                labeled_prompt_idx is not None
                and labeled_prompt_idx.get(ds) is not None
            ):
                labeled_train_data_dict[ds] = labeled_prompt_idx[ds]
            else:
                labeled_train_data_dict[ds] = range(len(data_dict[labeled_prefix][ds]))
        # Test data prompt indices.
        test_dict = {}
        for ds in eval_datasets:
            if test_prompt_idx is not None and test_prompt_idx.get(ds) is not None:
                test_dict[ds] = test_prompt_idx[ds]
            else:
                test_dict[ds] = range(len(data_dict[test_prefix][ds]))

        n_components = 1 if method == "TPC" else -1

        method_str = get_method_str(method)
        full_method_str = maybe_append_project_suffix(
            method_str, project_along_mean_diff
        )
        method_ = method
        if method.startswith("RCCS"):
            raise NotImplementedError()
            # method_ = "CCS"
            # if method != "RCCS0":
            #     # TODO: load constraints and old_biases from the correct directory for
            #     # RCCS.
            #     prev_rccs_params_dir = None
            #     # TODO: replace this with call to parameter-loading function
            #     # and use a different directory where the previous CCS params
            #     # are stored.
            #     constraints = np.load(os.path.join(prev_rccs_params_dir, "coef.npy"))
            #     old_biases = np.load(
            #         os.path.join(prev_rccs_params_dir, "intercept.npy")
            #     )
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
            weight_decay=_config["weight_decay"],
            opt=_config["opt"],
            num_orthogonal_directions=_config["num_orthogonal_directions"],
            span_dirs_combination=_config["span_dirs_combination"],
            log_reg=_config["log_reg"],
            pseudolabel=_config["pseudolabel"],
        )
        kwargs = dict(
            train_data_dict=train_data_dict,
            permutation_dict=permutation_dict,
            test_dict=test_dict,
            projection_dict=projection_dict,
            mode=mode,
            labeled_train_data_dict=labeled_train_data_dict,
            labeled_train_prefix=labeled_prefix,
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
                labeled_prefix=labeled_prefix,
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
                projected_sgd=_config["projected_sgd"],
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
                if not isinstance(fit_result, (list, tuple)):
                    fit_result = [fit_result]
                fit_result = [
                    {
                        k: v
                        for k, v in fr.items()
                        if k not in ("best_probe", "all_probes")
                    }
                    for fr in fit_result
                ]
                if len(fit_result) == 1:
                    fit_result = fit_result[0]
                save_fit_result(fit_result, run_dir, method, logger=_log)
            if _config["save_fit_plots"]:
                save_fit_plots(fit_result, run_dir, method, logger=_log)

        # Save parameters if needed.
        if _config["save_params"] and not _config["eval_only"]:
            save_params_dir = get_params_dir(
                run_dir,
                maybe_append_project_suffix(method_str, project_along_mean_diff),
                prefix,
                labeled_prefix=labeled_prefix,
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
            elif isinstance(classifier, ContrastPairClassifier):
                coef, bias = classifier.coef, classifier.bias
                coef = coef.detach().cpu().numpy()
                if bias is not None:
                    bias = bias.detach().cpu().numpy()
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
        if method in [
            "CCS+LR",
            "CCS-in-LR-span",
            "CCS+LR-in-span",
            "CCS-select-LR",
            "pseudolabel",
        ]:
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
                    "labeled_train_prefix": labeled_prefix,
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

                if method in [
                    "CCS+LR",
                    "CCS-in-LR-span",
                    "CCS+LR-in-span",
                    "CCS-select-LR",
                    "pseudolabel",
                ]:
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

            # If the eval file already exists, check if the headers are the
            # same. If they are, append to the file without adding new headers.
            # Otherwise, save to a new eval file and issue a warning.
            to_csv_kwargs = {"index": False}
            if os.path.exists(eval_results_path):
                existing_eval_results_df = pd.read_csv(eval_results_path)
                if set(ds_eval_results_df.columns) != set(
                    existing_eval_results_df.columns
                ):
                    warnings.warn(
                        f"Columns in the existing eval results file ({eval_results_path}) "
                        "do not match the current eval results. Saving to a new file."
                    )
                    eval_results_path = load_utils.get_eval_results_path(
                        run_dir, ds, append_time=True
                    )
                else:
                    to_csv_kwargs = {"mode": "a", "header": False}

            ds_eval_results_df.to_csv(eval_results_path, **to_csv_kwargs)

    expected_config_subset = {
        "datasets": train_datasets,
        "labeled_datasets": labeled_train_datasets,
        "eval_datasets": eval_datasets,
        "prefix": prefix,
        "labeled_prefix": labeled_prefix,
        "test_prefix": test_prefix,
    }
    if not _check_config_subset_unmodified(_config, expected_config_subset):
        config_subset = {key: _config[key] for key in expected_config_subset}
        raise ValueError(
            "The following config subset has been modified from the original config: \n"
            f"Current: {expected_config_subset}\n"
            f"Original: {config_subset}"
        )
