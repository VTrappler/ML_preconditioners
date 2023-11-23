import argparse
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf

sys.path.append("..")
from prec_models import construct_model_class
from prec_models.models_spectral import SVDConvolutional

model_list = [SVDConvolutional]
model_classes = {cl.__name__: cl for cl in model_list}

plt.style.use("seaborn-v0_8")
plt.set_cmap("magma")
sys.path.append("/home/GNlearning/")  # TODO: to robustify
# fig_folder = os.path.join(os.sep, "home", "lorenz", "artifacts") # TODO: to robustify
fig_folder = os.path.join(
    os.sep, "home", "GNlearning", "lorenz", "artifacts"
)  # TODO: to robustify

import tqdm


def bmm(a, b):
    return np.einsum("Bij,Bjk ->Bik", a, b)


def bt(a):
    return np.einsum("Bij->Bji", a)


def bouter(a, b):
    return np.einsum("Bnr,Bnr -> Bnn", a, b)


def bouter1D(a, b):
    return np.einsum("Bi,Bj -> Bij", a, b)


def bqr(a):
    Q = np.empty_like(a)
    for i, mat in enumerate(a):
        Q[i, ...] = np.linalg.qr(mat)[0]
    return Q


def sumLMP_ML(x_):
    pred = loaded_model.predict(x_)

    vecs, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    Ur = bqr(vecs)
    n = Ur.shape[1]
    r = Ur.shape[-1]
    acc = np.zeros((len(pred), n, n))
    print(f"{Sr.shape=}")
    for i in range(r):
        uu = Ur[..., i].reshape((len(pred), n, 1))
        uuprime = bmm(uu, bt(uu))
        acc += ((1 - Sr[:, i] ** (-1))).reshape(len(pred), 1, 1) * uuprime
    prec = np.eye(n).reshape(1, n, n) - acc
    mats = bmm(Ur, (Sr[..., None] * bt(Ur)))
    return mats, prec


def construct_LMP(S: np.ndarray, AS: np.ndarray, shift: float = 1.0) -> np.ndarray:
    print(f"{S.shape=}")
    In = np.eye(S.shape[1]).reshape(1, S.shape[1], S.shape[1])
    StASm1 = np.linalg.inv(bmm(bt(S), AS))
    left = In - bmm(bmm(S, StASm1), bt(AS))
    mid = In - bmm(bmm(AS, StASm1), bt(S))
    right = bmm(bmm(S, StASm1), bt(S))
    H = bmm(left, mid) + shift * right
    return H


def construct_invLMP(S: np.ndarray, AS: np.ndarray, shift: float = 1.0) -> np.ndarray:
    print(f"{S.shape=}")

    In = np.eye(S.shape[1]).reshape(1, S.shape[1], S.shape[1])
    StASm1 = np.linalg.inv(bmm(bt(S), AS))
    B = (
        In
        + (1 / shift) * bmm(bmm(AS, StASm1), bt(AS))
        - bmm(
            bmm(S, np.linalg.inv(bmm(bt(S), S))),
            bt(S),
        )
    )
    return B


# def construct_matrices(x_):
#     outputs = loaded_model.predict(np.asarray(x_).astype("f"))
#     S, AS = outputs[..., 0], outputs[..., 1]
#     return construct_invLMP(S, AS), construct_LMP(S, AS)


def construct_svd_ML(loaded_model, x_: np.ndarray):
    pred = loaded_model.predict(x_)
    Ur, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    Ur = bqr(Ur)
    # proj = bmm(qi, (inv_sv[..., None] * bt(qi)))
    return Sr.squeeze(), Ur.squeeze()


def construct_svd_ML(loaded_model, x_: np.ndarray):
    pred = loaded_model.predict(x_)
    Ur, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    Ur = bqr(Ur)
    # proj = bmm(qi, (inv_sv[..., None] * bt(qi)))
    return Sr.squeeze(), Ur.squeeze()


def construct_matrices(x_: np.ndarray):
    Sr, Ur = construct_svd_ML(loaded_model, x_)
    Sr_minus_1 = Sr ** (-1) - 1
    # H = construct_matrix_USUt(qi, eli)
    lr_approximation = bmm(Ur, (Sr[..., None] * bt(Ur)))
    prec = bmm(Ur, (Sr_minus_1[..., None] * bt(Ur))) + np.eye(Ur.shape[1])
    return lr_approximation, prec


def get_approximate_singular_val(x_: np.ndarray):
    pred = loaded_model.predict(x_)
    logsvals = pred[:, -1, :]
    sv = np.exp(logsvals)
    return sv


def gauss_newton_approximation_svd(
    x_, tlm_, indices, figname=os.path.join(fig_folder, "svd_approximation")
):
    for j, idx in enumerate(indices):
        plt.subplot(3, 5, 1 + j)
        tl = tlm_[idx, ...]
        gn = tl.T @ tl
        plt.imshow(gn)
        plt.title(f"Data: {idx}")
        plt.subplot(3, 5, 11 + j)
        plt.title(f"Approx: {idx}")
        plt.imshow(approximations[idx, ...])
        plt.subplot(3, 5, 6 + j)
        _, sv, _ = np.linalg.svd(gn)
        plt.plot(sv)
        approx_sv = np.sort(
            get_approximate_singular_val(x_[idx].reshape(-1, len(sv))).flatten()
        )[::-1]
        plt.plot(approx_sv, color="red")
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def sanity_check(
    approximations,
    preconditioners,
    indices,
    figname=os.path.join(fig_folder, "sanity_check"),
):
    product = preconditioners @ approximations
    for j, idx in enumerate(indices):
        plt.subplot(2, 5, 1 + j)
        plt.imshow(product[idx, ...])
        plt.title(f"Product: {idx}")
        plt.subplot(2, 5, 6 + j)
        _, sv, _ = np.linalg.svd(product[idx, ...])
        plt.plot(sv)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def preconditioned_svd(
    preconditioners,
    tlm_,
    indices,
    rank,
    figname=os.path.join(fig_folder, "preconditioning"),
):
    plt.figure(figsize=(10, 6))
    for j, idx in enumerate(indices):
        plt.subplot(2, 5, 1 + j)
        tl = tlm_[idx, ...]
        gn = tl.T @ tl
        preconditioned_gn = preconditioners[idx, ...] @ gn
        plt.imshow(preconditioned_gn)
        plt.subplot(2, 5, 6 + j)
        _, sv, _ = np.linalg.svd(preconditioned_gn)
        plt.plot(np.roll(sv, rank), label="prec")
        _, sv_original, _ = np.linalg.svd(gn)
        plt.plot(sv_original, label="original")
        plt.title(
            f"k: {np.linalg.cond(gn):.2e}\nk_p = {np.linalg.cond(preconditioned_gn):.2e}"
        )
        plt.yscale("log")
        plt.legend()
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def condition_numbers(
    preconditioners, tlm_, figname=os.path.join(fig_folder, "condition_numbers")
):
    original_condition_number = np.linalg.cond((bt(tlm_) @ tlm_))
    sorted_indices = np.argsort(original_condition_number)
    prec_condition_number = np.linalg.cond(preconditioners @ (bt(tlm_) @ tlm_))
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "condition": original_condition_number,
                    "matrix": "baseline",
                }
            ),
            pd.DataFrame(
                {
                    "condition": prec_condition_number,
                    "matrix": "prec",
                }
            ),
        ]
    )
    plt.subplot(2, 1, 1)
    plt.plot(original_condition_number[sorted_indices], ".")
    plt.plot(prec_condition_number[sorted_indices], ".")
    plt.yscale("log")
    plt.subplot(2, 1, 2)
    sns.boxplot(df, y="matrix", x="condition")
    plt.xscale("log")
    plt.savefig(figname)
    plt.close()


def range_singular_values(tlm_):
    cmin = np.inf
    cmax = -np.inf
    for i in tqdm.trange(len(tlm_)):
        tl = tlm_[i, ...]
        sv = np.linalg.svd(tl.T @ tl)[1]
        if sv.max() > cmax:
            cmax = sv.max()
        if sv.min() < cmin:
            cmin = sv.min()
    print(f"min: {cmin}, max: {cmax}")


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug("debug")
    parser = argparse.ArgumentParser(description="Use Surrogate in inference")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--run-id-yaml", type=str, default="")
    args = parser.parse_args()
    print(os.getenv("MLFLOW_TRACKING_URI"))
    if args.run_id:
        run_id = args.run_id
        data_path = "/root/raw_data/data.pkl"
    else:
        import yaml

        with open(args.run_id_yaml, "r") as fstream:
            run_id_yaml = yaml.safe_load(fstream)
            run_id = run_id_yaml["run_id"]
            data_path = run_id_yaml["data_path"]
            model_path = run_id_yaml["model_path"]
            print(f"{model_path=}")
    try:
        with open("config.yaml", "r") as fstream:
            config = yaml.safe_load(fstream)
            rank = config["architecture"]["rank"]
            window = config["model"]["window"]
            dim = config["model"]["dimension"]
            nsamples = config["data"]["nsamples"]
    except:
        rank = 0

    config = OmegaConf.load(
        os.path.join(os.sep, *model_path.split("/")[:-1], "config.yaml")
    )

    torch_model = construct_model_class(
        model_classes[config["architecture"]["class"]],
        rank=config["architecture"]["rank"],
    )
    state_dimension = config["model"]["dimension"]
    loaded_model = torch_model(
        state_dimension=state_dimension, config=config["architecture"]
    )

    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("loaded model")
    # logged_model = f"runs:/{run_id}/smoke_model"
    # # mlflow.pyfunc.get_model_dependencies(logged_model)
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    # logger.info(f"{loaded_model=}")

    # with open(data_path, "rb") as handle:
    #     data = pickle.load(handle)
    # x_, fo_, tlm_ = zip(*data)
    # data_path = "/root/raw_data/data_100_large"
    data_path = "/home/data/data_data_assimilation/data_100_large"  # TODO: to robustify
    x_mmap = os.path.join(data_path, "x.memmap")
    tlm_mmap = os.path.join(data_path, "tlm.memmap")

    x_ = np.memmap(
        x_mmap,
        dtype="float32",
        mode="c",
        shape=(nsamples, dim),
    )
    tlm_ = np.memmap(
        tlm_mmap,
        dtype="float32",
        mode="c",
        shape=(nsamples, dim * window, dim),
    )

    x_ = np.asarray(x_)
    # approximations, preconditioners = construct_matrices(x_)
    approximations, preconditioners = construct_matrices(x_)
    tlm_ = np.asarray(tlm_)
    # plt.figure(figsize=(12, 6))
    indices = np.random.randint(0, len(x_), size=5)
    # indices = [0, 1, 2, 3, 4]

    range_singular_values(tlm_)
    logger.info("svd approximation")
    gauss_newton_approximation_svd(
        x_, tlm_, indices, figname=os.path.join(fig_folder, "svd_approximation")
    )
    logger.info("preconditioning")
    preconditioned_svd(
        preconditioners,
        tlm_,
        indices,
        rank,
        figname=os.path.join(fig_folder, "preconditioning"),
    )

    logger.info("condition numbers")
    condition_numbers(
        preconditioners, tlm_, figname=os.path.join(fig_folder, "condition_numbers")
    )

    logger.info("sanity check")
    sanity_check(
        approximations,
        preconditioners,
        indices,
        figname=os.path.join(fig_folder, "sanity_check"),
    )
