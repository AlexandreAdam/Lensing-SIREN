from train_maml_siren import main
import os
from datetime import datetime
import pandas as pd
import copy
import numpy as np

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

DATE = datetime.now().strftime("%y%m%d%H%M%S")
BASEPATH = os.getenv("LSIREN_PATH")

HPARAMS = [
    "first_omega",
    "hidden_omega",
    "hidden_layers",
    "hidden_features",
    "learning_rate",
    "step_size",
    "loss_type",
    "lr_type",
    "batch_size",
    "num_adaptation_steps"
]

PARAMS_NICKNAME = {
    "first_omega": "FO",
    "hidden_omega": "HO",
    "hidden_layers": "HL",
    "hidden_features": "HF",
    "learning_rate": "lr",
    "step_size": "S",
    "loss_type": "loss",
    "lr_type": "lr_type",
    "batch_size": "B",
    "num_adaptation_steps": "TS"
}


def single_instance_args_generator(args):
    """

    Args:
        args: Namespace of argument parser

    Returns: A modified deep copy generator of Namespace to be fed to main of train_rim_unet.py

    """
    if args.strategy == "uniform":
        return uniform_grid_search(args)
    elif args.strategy == "exhaustive":
        return exhaustive_grid_search(args)
    else:
        raise NotImplementedError(f"{args.strategy} not in ['uniform', 'exhaustive']")


def uniform_grid_search(args):
    for gridsearch_id in range(1, args.n_models + 1):
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        nicknames = []
        params = []
        for p in HPARAMS :
            if isinstance(args_dict[p], list):
                if len(args_dict[p]) > 1:
                    # this way, numpy does not cast int to int64 or float to float32
                    args_dict[p] = args_dict[p][np.random.choice(range(len(args_dict[p])))]
                    nicknames.append(PARAMS_NICKNAME[p])
                    params.append(args_dict[p])
                else:
                    args_dict[p] = args_dict[p][0]
        param_str = "_" + "_".join([f"{nickname}{param}" for nickname, param in zip(nicknames, params)])
        args_dict.update({"logname": args.model_id + "_" + f"{gridsearch_id:03d}" + param_str + "_" + DATE})
        yield new_args


def exhaustive_grid_search(args):
    """
    Lexicographic ordering of given parameter lists, up to n_models deep.
    """
    from itertools import product
    grid_params = []
    for p in HPARAMS:
        if isinstance(vars(args)[p], list):
            if len(vars(args)[p]) > 1:
                grid_params.append(vars(args)[p])
    lexicographically_ordered_grid_params = product(*grid_params)
    for gridsearch_id, lex in enumerate(lexicographically_ordered_grid_params):
        if gridsearch_id >= args.n_models:
            return
        new_args = copy.deepcopy(args)
        args_dict = vars(new_args)
        nicknames = []
        params = []
        i = 0
        for p in HPARAMS:
            if isinstance(args_dict[p], list):
                if len(args_dict[p]) > 1:
                    args_dict[p] = lex[i]
                    i += 1
                    nicknames.append(PARAMS_NICKNAME[p])
                    params.append(args_dict[p])
                else:
                    args_dict[p] = args_dict[p][0]
        param_str = "_" + "_".join([f"{nickname}{param}" for nickname, param in zip(nicknames, params)])
        args_dict.update({"model_id": args.model_id + "_" + f"{gridsearch_id:03d}" + param_str + "_" + DATE})
        yield new_args


def distributed_strategy(args):
    gridsearch_args = list(single_instance_args_generator(args))
    experiment_name = args.model_id
    for gridsearch_id in range((THIS_WORKER - 1), len(gridsearch_args), N_WORKERS):
        run_args = gridsearch_args[gridsearch_id]
        history = main(run_args)
        params_dict = {k: v for k, v in vars(run_args).items() if k in HPARAMS}
        params_dict.update({
            "experiment_id": run_args.model_id,
            "loss": history["loss"][-1],
        })
        # Save hyperparameters and scores in shared csv for this gridsearch
        df = pd.DataFrame(params_dict, index=[gridsearch_id])
        grid_csv_path = os.path.join(BASEPATH, "results", f"{experiment_name}.csv")
        this_run_csv_path = os.path.join(BASEPATH, "results", f"{run_args.model_id}.csv")
        if not os.path.exists(grid_csv_path):
            mode = "w"
            header = True
        else:
            mode = "a"
            header = False
        df.to_csv(grid_csv_path, header=header, mode=mode)
        pd.DataFrame(history).to_csv(this_run_csv_path)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="maml_siren_grid")
    parser.add_argument("--strategy", default="exhaustive")
    parser.add_argument("--n_models", default=1, type=int)
    parser.add_argument("--dataset", default="hkappa188hst_TNG100_rau_spl.h5")
    parser.add_argument("--data_len", default=None, type=int)
    parser.add_argument("--epochs",  default=50, type=int)
    parser.add_argument("--first_omega", default=0.5, nargs="+", type=float)
    parser.add_argument("--hidden_omega", default=0.5, nargs="+", type=float)
    parser.add_argument("--hidden_layers", default=2, nargs="+", type=int, help="Number of SIREN hidden layers")
    parser.add_argument("--hidden_features", default=50, nargs="+", type=int, help="Number of SIREN hidden feature per layers")
    parser.add_argument("--learning_rate", default=5e-3, type=float, nargs="+", help="Outer loop learning rate")
    parser.add_argument("--step_size", default=0.05, nargs="+", type=float, help="Inner loop learning rate")
    parser.add_argument("--loss_type", default="image", help="'image', 'gradient', 'laplace', 'image_gradient', 'image_laplace', 'gradient_laplace' or 'image_gradient_laplace'")
    parser.add_argument("--lr_type", default="static", nargs="+", help="'static' (default), 'global', 'per_step', 'per_parameter', 'per_parameter_per_step'")
    parser.add_argument("--num_adaptation_steps", default=5, nargs="+", type=int, help="inner optimization steps")
    parser.add_argument("--batch_size", default=2, type=int, help="Number of tasks in the inner loop")
    parser.add_argument("--epochs_til_checkpoint", default=2)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--max_time", default=np.inf, type=float)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    distributed_strategy(args)