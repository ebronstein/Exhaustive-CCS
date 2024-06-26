import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

from utils_extraction.func_utils import eval_adder, getAvg
from utils_extraction.load_utils import (
    get_params_dir,
    get_probs_save_path,
    get_zeros_acc,
    load_hidden_states_for_datasets,
    maybe_append_project_suffix,
    save_params,
)
from utils_extraction.method_utils import is_method_unsupervised, mainResults
from utils_generation import hf_utils
from utils_generation import parser as parser_utils
from utils_generation.hf_auth_token import HF_AUTH_TOKEN
from utils_generation.save_utils import get_model_short_name, get_results_save_path

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_prefix = global_dict["registered_prefix"]


parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str)
parser.add_argument("--prefix", nargs="+", default = ["normal"], choices = registered_prefix)
parser.add_argument("--datasets", nargs="+", default = registered_dataset_list)
parser.add_argument("--test", type = str, default = "testall", choices = ["testone", "testall"])
parser.add_argument("--data_num", type = int, default = 1000)
parser.add_argument("--method_list", nargs="+", default = ["0-shot", "TPC", "KMeans", "LR", "BSS", "CCS"], help=(
        "The name of the method, which should either be in {0-shot, TPC, KMeans, LR, BSS, CCS, Random}\n"
        "or be of the form RCCSi, where i is an integer: to run 10 iteration of RCCS, pass RCCS0, ..., RCCS9 as argument "
        "(it should start by RCCS0). Stats will be saved for each iterations as a separate experiment, "
        "and the concatenation of probes' parameters will be saved as if you had run a method named `RCCS`.."
    )
)
parser.add_argument("--mode", type = str, default = "auto", choices = ["auto", "minus", "concat"], help = "How you combine h^+ and h^-.")
parser.add_argument("--save_dir", type = str, default = "extraction_results", help = "where the csv and params are saved")
parser.add_argument("--append", action="store_true", help = "Whether to append content in frame rather than rewrite.")
parser.add_argument("--overwrite", action="store_true", help = "Whether to overwrite the existing results in `save_dir`.")
parser.add_argument("--load_dir", type = str, default = "generation_results", help = "Where the hidden states and zero-shot accuracy are loaded.")
parser.add_argument("--load_classifier", action="store_true", help = "Whether to load the classifier.")
parser.add_argument("--params_load_dir", type = str, default = "extraction_results", help = "Directory from which the classifier model's parameters are loaded.")
parser.add_argument("--location", type = str, default = "auto")
parser.add_argument("--layer", type = int, default = -1)
parser.add_argument("--zero", type = str, default = "generation_results")
parser.add_argument("--seed", type = int, default = 0)
parser.add_argument("--prompt_save_level", default = "all", choices = ["single", "all"])
parser.add_argument("--save_states", action="store_true", help="Whether to save the p0, p1, labels.")
parser.add_argument("--save_params", dest="save_params", action="store_true", default=True, help="Whether to save the parameters.")
parser.add_argument("--no_save_params", dest="save_params", action="store_false", help="Whether to save the parameters.")
parser.add_argument("--no_save_results", action="store_true", help="Whether to save the results in a CSV file.")
parser.add_argument("--test_on_train", action="store_true", help="Whether to test on the train set.")
parser.add_argument(
    "--project_along_mean_diff",
    action="store_true",
    help="Whether to project the data along the difference of means. You can also use the suffix -md in the method name.",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Whether to print more verbose results.")
args = parser.parse_args()

dataset_list = args.datasets
assert args.test != "testone", NotImplementedError("Current extraction program does not support applying method on prompt-specific level.")

args.location = parser_utils.get_states_location_str(args.location, args.model, use_auth_token=HF_AUTH_TOKEN)
num_layers = hf_utils.get_num_hidden_layers(args.model)
# TODO: validate args.location more extensively.
if args.location == "decoder" and args.layer < 0:
    args.layer += num_layers


def print_args(args):
    print("-------- args --------")
    for key in list(vars(args).keys()):
        print("{}: {}".format(key, vars(args)[key]))
    print("-------- args --------")


def methodHasLoss(method):
    return method in ["BSS", "CCS"] or method.startswith("RCCS")


def saveCsv(csv, save_dir, model_str, prefix, seed, msg = ""):
    model_short_name = get_model_short_name(model_str)
    dir = os.path.join(save_dir, "{}_{}_{}.csv".format(model_short_name, prefix, seed))
    csv.to_csv(dir, index = False)
    print("{} Saving to {} at {}".format(msg, dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == "__main__":
    # check the os existence
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.save_params and not os.path.exists(os.path.join(args.save_dir, "params")):
        os.makedirs(os.path.join(args.save_dir, "params"), exist_ok=True)

    print_args(args)

    # each loop will generate a csv file
    for global_prefix in args.prefix:
        print("---------------- model = {}, prefix = {} ----------------".format(args.model, global_prefix) )
        # Set the random seed, in which case the permutation_dict will be the same
        random.seed(args.seed)
        np.random.seed(args.seed)

        # shorten the name
        model = args.model
        data_num = args.data_num

        # Start calculate numbers
        # std is over all prompts within this dataset
        results_filepath = get_results_save_path(args.save_dir, args.model, global_prefix, args.seed)
        if os.path.exists(results_filepath) and not args.overwrite:
            raise ValueError("Results {} already exists. Please use --overwrite to overwrite it.".format(results_filepath))
        elif os.path.exists(results_filepath) and args.append:
            csv = pd.read_csv(results_filepath)
            print("Loaded {} at {}".format(results_filepath, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        else:
            csv = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])

        if "0-shot" in args.method_list:
            # load zero-shot performance
            rawzeros = pd.read_csv(os.path.join(args.load_dir, "{}.csv".format(args.zero)))
            # Get the global zero acc dict (setname, [acc])
            zeros_acc = get_zeros_acc(
                args.load_dir,
                csv_name = args.zero,
                mdl_name = model,
                dataset_list = dataset_list,
                prefix = global_prefix,
            )
            for setname in dataset_list:
                if args.prompt_save_level == "all":
                    csv = eval_adder(csv, model, global_prefix, "0-shot", "", "", setname,
                            np.mean(zeros_acc[setname]),
                            np.std(zeros_acc[setname]),"","","","")
                else:   # For each prompt, save one line
                    for idx in range(len(zeros_acc[setname])):
                        csv = eval_adder(csv, model, global_prefix, "0-shot",
                                    prompt_level= idx, train= "", test = setname,
                                    accuracy = zeros_acc[setname][idx],
                                    std = "", ece="", location = "", layer = "", loss = "",
                                    sim_loss = "", cons_loss = "")


            if not args.no_save_results:
                saveCsv(csv, args.save_dir, args.model, global_prefix, args.seed, "After calculating zeroshot performance.")

        for method in args.method_list:
            if method == "0-shot":
                continue
            print("-------- method = {} --------".format(method))

            project_along_mean_diff = "-md" in method or args.project_along_mean_diff
            if project_along_mean_diff:
                method = method.replace("-md", "")



            method_use_concat = (method in {"CCS", "Random"}) or method.startswith("RCCS")

            mode = args.mode if args.mode != "auto" else ("concat" if method_use_concat else "minus")
            # load the data_dict and permutation_dict
            data_dict, permutation_dict = load_hidden_states_for_datasets(
                args.load_dir,
                mdl_name= model,
                dataset_list=dataset_list,
                prefix = global_prefix,
                location = args.location,
                layer = args.layer,
                mode = mode
            )

            test_dict = {key: range(len(data_dict[key])) for key in dataset_list}

            for train_set in ["all"] + dataset_list:

                train_list = dataset_list if train_set == "all" else [train_set]
                projection_dict = {key: range(len(data_dict[key])) for key in train_list}

                n_components = 1 if method == "TPC" else -1

                save_file_prefix = (
                    get_probs_save_path(args.save_dir, args.model, method, project_along_mean_diff, args.seed, train_set)
                    if args.save_states
                    else None
                )

                method_str = "CCS" if method.startswith("RCCS") else method
                params_file_name = get_params_dir(
                    model,
                    global_prefix,
                    maybe_append_project_suffix(method_str, project_along_mean_diff),
                    train_set,
                    args.seed,
                )

                method_ = method
                constraints = None
                if method.startswith("RCCS"):
                    method_ = "CCS"
                    if method != "RCCS0":
                        # TODO: replace this with call to parameter-loading function.
                        constraints = np.load(
                            os.path.join(args.save_dir, "params", "coef_{}.npy".format(params_file_name))
                        )
                        old_biases = np.load(
                            os.path.join(args.save_dir, "params", "intercept_{}.npy".format(params_file_name))
                        )

                load_classifier_dir_and_name = (args.params_load_dir, params_file_name) if args.load_classifier else None

                # return a dict with the same shape as test_dict
                # for each key test_dict[key] is a unitary list
                res, lss, ece, pmodel, cmodel = mainResults(
                    data_dict = data_dict,
                    permutation_dict = permutation_dict,
                    projection_dict = projection_dict,
                    test_dict = test_dict,
                    projection_method = "PCA",
                    n_components = n_components,
                    load_classifier_dir_and_name=load_classifier_dir_and_name,
                    classification_method = method_,
                    print_more=args.verbose,
                    save_file_prefix=save_file_prefix,
                    test_on_train=args.test_on_train,
                    constraints=constraints,
                    project_along_mean_diff=project_along_mean_diff,
                )

                # save params except for KMeans
                if method in ["TPC", "BSS", "CCS", "Random", "LR"]:
                    if method in ["TPC", "BSS"]:
                        coef, bias = cmodel.coef_ @ pmodel.getDirection(), cmodel.intercept_
                    elif method in ["CCS", "Random"]:
                        coef = cmodel.coef
                        bias = cmodel.bias
                    elif method == "LR":
                        coef, bias = cmodel.coef_, cmodel.intercept_
                    else:
                        assert False
                    if args.save_params:
                        save_params(
                            args.save_dir,
                            params_file_name,
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
                    if args.save_params:
                        save_params(args.save_dir, params_file_name, coef, bias)

                acc, std, loss, sim_loss, cons_loss = getAvg(res), np.mean([np.std(lis) for lis in res.values()]), *np.mean([np.mean(lis, axis=0) for lis in lss.values()], axis=0)
                ece_dict = {key: [ece[0] for ece in ece_vals] for key, ece_vals in ece.items()}
                ece_flip_dict = {key: [ece[1] for ece in ece_vals] for key, ece_vals in ece.items()}
                mean_ece = getAvg(ece_dict)
                mean_ece_flip = getAvg(ece_flip_dict)
                print("method = {:8}, prompt_level = {:8}, train_set = {:10}, avgacc is {:.2f}, std is {:.2f}, loss is {:.4f}, sim_loss is {:.4f}, cons_loss is {:.4f}, ECE is {:.4f}, ECE (1-p) is {:.4f}".format(
                    maybe_append_project_suffix(method, project_along_mean_diff), "all", train_set, 100 * acc, 100 * std, loss, sim_loss, cons_loss, mean_ece, mean_ece_flip)
                )

                for key in dataset_list:
                    if args.prompt_save_level == "all":
                        loss, sim_loss, cons_loss = np.mean(lss[key], axis=0) if methodHasLoss(method) else ("", "", "")
                        csv = eval_adder(csv, model, global_prefix, maybe_append_project_suffix(method, project_along_mean_diff), "all", train_set, key,
                                    accuracy = np.mean(res[key]),
                                    std = np.std(res[key]),
                                    ece=np.mean(ece_dict[key]),
                                    ece_flip=np.mean(ece_flip_dict[key]),
                                    location = args.location,
                                    layer = args.layer,
                                    loss = loss, sim_loss = sim_loss, cons_loss = cons_loss,
                                    )
                    else:
                        for idx in range(len(res[key])):
                            loss, sim_loss, cons_loss = lss[key][idx] if methodHasLoss(method) else ("", "", "")
                            csv = eval_adder(csv, model, global_prefix, maybe_append_project_suffix(method, project_along_mean_diff), idx, train_set, key,
                                        accuracy = res[key][idx],
                                        std = "",
                                        ece=ece_dict[key][idx],
                                        ece_flip=ece_flip_dict[key][idx],
                                        location = args.location,
                                        layer = args.layer,
                                        loss = loss, sim_loss = sim_loss, cons_loss = cons_loss,
                                        )

        if not args.no_save_results:
            saveCsv(csv, args.save_dir, args.model, global_prefix, args.seed, "After finish {}".format(maybe_append_project_suffix(method, False)))
