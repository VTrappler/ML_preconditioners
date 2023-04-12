import os
import sys
import pickle

sys.path.append("..")

from data import TangentLinearDataModule
from lorenz_wrapper import LorenzWrapper
from prec_models.__models import (
    LimitedMemoryPrecLinearOperator,
    construct_model_class,
    LimitedMemoryPrec,
    LimitedMemoryPrecRegularized,
    LimitedMemoryPrecVectorNorm,
    LimitedMemoryPrecSym,
    DeflationPrec,
    SVDPrec,
)
from models_unstructured import BandMatrix, LowRank
import pandas as pd

n_obs = 10


def choose_pytorch_model(ray_folder, rank):
    if ray_folder == "LimitedMemoryPrec":
        torch_model = construct_model_class(LimitedMemoryPrec, rank=rank, AS=True)
    elif ray_folder == "LMPsymmetric":
        torch_model = construct_model_class(LimitedMemoryPrecSym, rank=rank, AS=True)
    elif ray_folder == "LMPlinearop":
        torch_model = construct_model_class(LimitedMemoryPrecLinearOperator, rank=rank)
    elif ray_folder == "LMPortho":
        torch_model = construct_model_class(
            LimitedMemoryPrecRegularized, rank=rank, AS=True
        )
    elif ray_folder == "LMPvecnorm":
        torch_model = construct_model_class(
            LimitedMemoryPrecVectorNorm, rank=rank, AS=True
        )
    elif ray_folder == "LMPvecnorm_rnd":
        torch_model = construct_model_class(
            LimitedMemoryPrecVectorNorm, rank=rank, AS=True
        )
    elif ray_folder == "LMPvecnorm_vecdata":
        torch_model = construct_model_class(
            LimitedMemoryPrecVectorNorm, rank=rank, AS=True
        )
    elif ray_folder == "lowrank":
        torch_model = construct_model_class(LowRank, rank=rank)
    elif ray_folder == "band":
        torch_model = construct_model_class(BandMatrix, rank=rank)
    elif ray_folder == "deflation":
        torch_model = construct_model_class(DeflationPrec, rank=rank)
    elif ray_folder == "LMPSVD":
        torch_model = construct_model_class(SVDPrec, rank=rank)

    return torch_model


def select_model_from_dataframe(state_dimension, rank, loss_type, ray_folder):
    xp = pd.read_csv("xp_tracker.csv")
    xp_row = xp.query(
        f'rank == {rank} and dim == {state_dimension} and loss_type == "{loss_type}" and ray_folder == "{ray_folder}" and n_obs == {n_obs}'
    )
    name = xp_row.name
    print(xp_row[["name", "loss_value", "n_epochs"]])
    name = xp_row.loc[xp_row["loss_value"].idxmin(), "name"]
    data_path = xp_row.obs_file.values[0]
    model_name = f"{state_dimension}-{loss_type}-rank-{rank}-{name}"
    # print(xp_row)
    torch_model = choose_pytorch_model(ray_folder, rank)
    print(f"--> {model_name}")
    model_path, config_path = (
        f"{ray_folder}/{model_name}/best_{model_name}.pth",
        f"{ray_folder}/{model_name}/{model_name}_config.yaml",
    )
    if not os.path.exists(config_path):
        config_path = f"{ray_folder}/{model_name}/{name}_hpo.yaml"
    return model_name, torch_model, model_path, config_path, data_path


def select_model_from_name(name, ray_folder):
    xp = pd.read_csv("xp_tracker.csv")
    xp_row = xp.query(f'name == "{name}"')
    print(xp_row[["name", "loss_value", "n_epochs"]])
    name = xp_row.name.values[0]
    state_dimension = xp_row.dim.values[0]
    rank = xp_row["rank"].values[0]
    loss = xp_row.loss_type.values[0]
    data_path = xp_row.obs_file.values[0]
    model_name = f"{state_dimension}-{loss}-rank-{rank}-{name}"
    torch_model = choose_pytorch_model(ray_folder, rank)
    print(f"--> {model_name}")
    model_path, config_path = (
        f"{ray_folder}/{model_name}/best_{model_name}.pth",
        f"{ray_folder}/{model_name}/{model_name}_config.yaml",
    )
    if not os.path.exists(config_path):
        config_path = f"{ray_folder}/{model_name}/{name}_hpo.yaml"
    return model_name, torch_model, model_path, config_path, data_path
