## System imports
import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

import shutil
from distutils.dir_util import copy_tree
import functools
import argparse
from datetime import datetime

## Config related imports
import randomname
import yaml
from omegaconf import DictConfig, OmegaConf

import pandas as pd


## ML imports
import torch
import pytorch_lightning as pl


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from prec_data.data import TangentLinearDataModule, LorenzTLVectorIterableDataModule

import prec_models.models as models
import prec_models.models_unstructured as models_unstructured

from models import (
    LimitedMemoryPrecLinearOperator,
    LimitedMemoryPrecRegularized,
    LimitedMemoryPrecSym,
    LimitedMemoryPrecVectorNorm,
    LimitedMemoryPrec,
)
from prec_models.base_models import construct_model_class



## HPO related imports
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from ray.tune.search.hyperopt import HyperOptSearch


cwd = os.getcwd()


def train_tune(
    config: dict,
    state_dimension: int,
    torch_model,
    data_path: str,
    num_epochs: int,
    seed: int,
    **kwargs: dict
):
    ## Load the model
    try:
        model = torch_model(state_dimension=state_dimension, config=config, AS=True) # If requires AS argument
    except TypeError:
        model = torch_model(state_dimension=state_dimension, config=config)
    torch.manual_seed(seed)

    ## Select datamodule (either fully loaded or iterable)
    if data_path is not None:
        datamodule = TangentLinearDataModule(
            path=data_path,
            batch_size=config["batch_size"],
            # splitting_lengths = [80_000, 10_000, 10_000],
            splitting_lengths=[0.8, 0.1, 0.1],
            shuffling=True,
            num_workers=4,
        )
    else:
        datamodule = LorenzTLVectorIterableDataModule(
            state_dimension=state_dimension,
            nobs=10,
            len_dataset=200,
            nvectors=0,
            batch_size=config["batch_size"],
            num_workers=1,
            persistent_workers=True,
            # splitting_lengths=[0.8, 0.1, 0.1],
            shuffling=True,
        )
    # print(config)
    
    ## Use a learning rate scheduler
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={"loss": "Loss/val_loss"},
                filename="checkpoint",
                on="validation_end",
            ),
            lr_monitor,
        ],
        logger=TensorBoardLogger(save_dir="", version="."),
        enable_progress_bar=False,
        log_every_n_steps=20
    )
    trainer.fit(model, datamodule)
    return model


def train_tune_asha(
    state_dimension,
    torch_model,
    rank,
    loss_type,
    data_path,
    dir,
    num_samples,
    num_epochs,
    concurrent_trials,
    gpus_per_trial=1,
    config_yaml="",
):
    
    ## HPO options
    config = {
        "n_layers": tune.choice([2, 3, 4, 5]),
        "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]),
        "neurons_per_layer": tune.choice([128, 256, 512]),
        "batch_size": tune.choice([16, 32]),
    }
    
    dir_path = os.path.join(cwd, dir, "")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    _existing_xps = os.listdir(dir_path)

    ## Generate experiment name
    _randomize_name = True
    while _randomize_name:
        rnd_name = randomname.get_name()
        name = f"{state_dimension}-{loss_type}-rank-{rank}-{rnd_name}"
        if name not in _existing_xps:
            break

    if data_path is not None:
        import re
        regex = r"\d+obs"
        try:
            n_obs = int(re.search(r"\d+", re.search(regex, data_path).group()).group())
        except AttributeError: 
            n_obs = 10
    else:
        n_obs = 10

    print(f"\nName: {name}\n\n")
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    ## ASHA Scheduler kill trainings not performing
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=5, reduction_factor=1.5)

    reporter = CLIReporter(
        parameter_columns=["n_layers", "neurons_per_layer", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"],
    )

    seed = hash(name)
    print(f"{data_path=}\n")
    print(f"{n_obs=}\n")
    print(f"{seed=}\n")
    
    config_filepath = os.path.join(dir_path, f"config_{rnd_name}.yaml")
    
    if len(config_yaml) == 0:
        dict_conf = {
            "archi": {
                "dim": state_dimension,
                "rank": rank,
                "loss": loss_type,
                "ray_folder": ray_folder,
                "data-path": data_path,
                "iterable_data": data_path is None,
                "model": ray_folder,
                "epochs": n_epochs,
                "samples": n_samples,
                "concurrent-trials": concurrent_trials
            },
            "notes": "",
        }
        conf = OmegaConf.create(dict_conf)
        with open(config_filepath, "w") as fp:
            OmegaConf.save(config=conf, f=fp.name)
    else:
        with open(f"{config_yaml}", "r") as fp:
            loaded = OmegaConf.load(fp.name)
        with open(config_filepath, "w") as fp:
            OmegaConf.save(config=loaded, f=fp.name)   

    train_tune_with_parameters = tune.with_parameters(
        train_tune,
        state_dimension=state_dimension,
        torch_model=torch_model,
        data_path=data_path,
        num_epochs=num_epochs,
        seed=seed,
    )

    hyperopt_search = HyperOptSearch(metric="loss", mode="min", n_initial_points=2)
    log_path = os.path.join(dir_path, f"{name}_log.txt")
    with open(log_path, "w") as f:
        f.write(f"{name}\n")
        f.write(f"{data_path}\n")
        f.write(f"{seed}\n")

    analysis = tune.run(
        train_tune_with_parameters,
        metric="loss",
        mode="min",
        config=config,
        search_alg=hyperopt_search,
        num_samples=num_samples,
        scheduler=scheduler,
        name=name,
        checkpoint_at_end=False,
        keep_checkpoints_num=5,
        # checkpoint_freq=10,
        local_dir=f"{dir}/",
        max_concurrent_trials=concurrent_trials,
        raise_on_failed_trial=True,
        log_to_file=True,
        verbose=3
    )
    ## Best trial output
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    print(f"{analysis.best_trial=}")  # Get best trial
    print(f"{analysis.best_config=}")  # Get best trial's hyperparameters
    print(f"{analysis.best_logdir=}")  # Get best trial's logdir
    print(f"{analysis.best_checkpoint=}")  # Get best trial's best checkpoint
    print(f"{analysis.best_result=}")  # Get best trial's last results
    print(f"{analysis.best_result_df=}")  # Get best result as pandas dataframe

    best_trained_model = torch_model(
        state_dimension=state_dimension, config=best_trial.config
    )
    # best_checkpoint_dir = best_trial.checkpoint.value
    with open(log_path, "a") as f:
        # f.write(f"{best_checkpoint_dir}\n")
        f.write("Best trial config: {}\n".format(best_trial.config))
        f.write(f"{analysis.best_trial=}\n")  # Get best trial
        f.write(f"{analysis.best_config=}\n")  # Get best trial's hyperparameters
        f.write(f"{analysis.best_logdir=}\n")  # Get best trial's logdir
        f.write(f"{analysis.best_result['loss']=}\n")  # Get best trial's last results
        f.write(f"{analysis.best_checkpoint=}\n")  # Get best trial's best checkpoint

    print(f"{analysis.best_checkpoint=}")
    print(f"{analysis.best_checkpoint._local_path=}")
    experiment_path = os.path.split(analysis.best_checkpoint._local_path)[0]

    named_exp_path = os.path.join(dir_path, f"{name}", "")
    
    print(f"Copy {experiment_path} to {named_exp_path}")
    copy_tree(experiment_path, os.path.join(named_exp_path, 'training_log'))

    load_ckpt = torch.load(os.path.join(analysis.best_checkpoint._local_path, "checkpoint"))
    best_trained_model.load_state_dict(load_ckpt["state_dict"])
    best_model_path = os.path.join(named_exp_path, f"best_{name}.pth")
    torch.save(
        {
            "model_state_dict": best_trained_model.state_dict(),
            # 'optimizer_state_dict': best_trained_model.optimizers().state_dict(),
            "config": best_trial.config,
        },
        best_model_path,
    )

    print(f"Best model saved in {best_model_path}")

    ## Custom and dirty experiment tracking using a custom CSV
    
    with open(os.path.join(named_exp_path, f"{rnd_name}_hpo.yaml"), "w") as yaml_file:
        yaml.dump(best_trial.config, yaml_file, default_flow_style=False)

    shutil.move(config_filepath, os.path.join(named_exp_path, f"config_{rnd_name}.yaml"))
    
    columns = [
        "name",
        "ray_folder",
        "loss_type",
        "loss_value",
        "dim",
        "rank",
        "obs_file",
        "n_obs",
        "seed",
        "n_epochs",
        "n_samples",
        "date",
    ]

    xp_data = [
        f"{rnd_name}",
        f"{dir}",
        f"{loss_type}",  # loss type
        f"{analysis.best_result['loss']}",
        f"{state_dimension}",
        f"{rank}",
        f"{data_path}",
        f"{n_obs}",
        f"{seed}",
        f"{num_epochs}",
        f"{num_samples}",
        f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
    ]

    add_data = pd.DataFrame([xp_data], columns=columns)
    with open("xp_tracker.csv", "a") as f:
        add_data.to_csv(f, header=False)
        
    
    # Clean useless folders
    rm_train_tune_folders(dir, name)


def rm_train_tune_folders(ray_folder, name):
    """ Remove temp directories created by raytune
    """
    dir_path = os.path.join(cwd, f"{ray_folder}", f"{name}")
    folders_paths = next(os.walk(dir_path)[1]
    print("Experiment paths")
    for fo in folders_paths:
        print(fo)
    train_folders_path = [fol for fol in folders_paths if "train_tune" in fol]
    for fold in train_folders_path:
        print(f"Removing {fold}")
        shutil.rmtree(
            os.path.join(dir_path, f"{fold}", ignore_errors=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dim", type=int, default=40, help="State dimension of the Lorenz model"
    )
    parser.add_argument("--rank", type=int, default=20, help="Rank")
    parser.add_argument("--loss", type=str, default="inv", help="Type of loss")
    parser.add_argument("--ray_folder", type=str, help="Ray folder", default="lowrank")
    parser.add_argument(
        "--data-path",
        type=str,
        help="pkl training file",
        default="/da_dev/GNlearning/data/training_id_40_20000_10obs.pkl",
    )
    parser.add_argument("--iterable-data", action="store_true")
    parser.add_argument("--conf", type=str, default="")
    parser.add_argument("--concurrent-trials", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model", type=str, help="Type of model to use", default="LMP")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=500)
    parser.add_argument(
        "--samples", type=int, help="Number of samples for HPO", default=25
    )
    config_yaml = ''
    args = parser.parse_args()
    if len(args.conf) > 0:
        conf = OmegaConf.load(args.conf)
        n_samples = conf.archi["samples"]
        n_epochs = conf.archi["epochs"]
        if conf.archi["iterable_data"]:
            data_path = None
            datatype = 'iterable'

        else:
            data_path = conf.archi["data-path"]
            datatype='full'

        model_str = conf.archi["model"]
        rk = conf.archi["rank"]
        state_dimension = conf.archi["dim"]
        loss_str = conf.archi["loss"]
        concurrent_trials = conf.archi["concurrent-trials"]
        ray_folder = conf.archi["ray_folder"]
        config_yaml = args.conf
    else:
        if args.smoke_test:
            n_samples = 2
            n_epochs = 5
        else:
            n_epochs = args.epochs
            n_samples = args.samples

        if args.iterable_data:
            data_path = None
            datatype = 'iterable'
        else:
            data_path = args.data_path
            datatype='full'
        print(f"{data_path=}")
        model_str = args.model
        rk = args.rank
        state_dimension = args.dim
        loss_str = args.loss
        concurrent_trials = args.concurrent_trials
        ray_folder = args.ray_folder

    if model_str == "LMP":
        model = construct_model_class(LimitedMemoryPrec, rank=rk, datatype=datatype)
    elif model_str == "LMPortho":
        model = construct_model_class(LimitedMemoryPrecRegularized, rank=rk, datatype=datatype)
    elif model_str == "LMPvecnorm":
        model = construct_model_class(LimitedMemoryPrecVectorNorm, rank=rk, datatype=datatype)
    elif model_str == "LMPsymmetric":
        model = construct_model_class(LimitedMemoryPrecSym, rank=rk, datatype=datatype)
    elif model_str == "LMPlinearop":
        model = construct_model_class(LimitedMemoryPrecLinearOperator, rank=rk, datatype=datatype)
    elif model_str == "band":
        model = construct_model_class(models_unstructured.BandMatrix, bw=rk)
    elif model_str == "lowrank":
        model = construct_model_class(models_unstructured.LowRank, rank=rk)
    elif model_str == "deflation":
        model = construct_model_class(models.DeflationPrec, rank=rk)
    elif model_str == "SVD":
        model = construct_model_class(models.SVDPrec, rank=rk, datatype=datatype)



    train_tune_asha(
        state_dimension=state_dimension,
        torch_model=model,
        rank=rk,
        loss_type=loss_str,
        data_path=data_path,
        dir=ray_folder,
        num_samples=n_samples,
        num_epochs=n_epochs,
        concurrent_trials=concurrent_trials,
        config_yaml=config_yaml
    )
