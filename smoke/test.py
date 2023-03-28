import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np

sys.path.append("/home/")
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
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    vecs, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    Ur = bqr(vecs)
    n = Ur.shape[1]
    r = Ur.shape[-1]
    acc = np.zeros((len(pred), n, n))
    print(f"{Sr.shape=}")
    for i in range(r):
        uu  = Ur[..., i].reshape((len(pred), n, 1))
        uuprime = bmm(uu, bt(uu))
        acc += ((1 - Sr[:, i]**(-1))).reshape(len(pred), 1, 1) * uuprime
    prec = np.eye(n).reshape(1, n, n) - acc
    mats = bmm(Ur, (Sr[..., None] * bt(Ur)))
    return mats, prec


def construct_matrices(x_):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    vecs, logsvals = pred[:, :-1, :], pred[:, -1, :]
    sv = np.exp(logsvals) - 1
    inv_sv = np.exp(-logsvals) - 1
    qi = bqr(vecs)
    # H = construct_matrix_USUt(qi, eli)
    mats = bmm(qi, (sv[..., None] * bt(qi))) + np.eye(vecs.shape[1])
    prec = bmm(qi, (inv_sv[..., None] * bt(qi))) + np.eye(vecs.shape[1])
    return mats, prec


def get_approximate_singular_val(x_):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    logsvals = pred[:, -1, :]
    sv = np.exp(logsvals)
    return sv


def gauss_newton_approximation_svd(
    x_, tlm_, indices, figname="/home/figures/inference_approx"
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
    approximations, preconditioners, indices, figname="/home/figures/sanity_check"
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
    preconditioners, tlm_, indices, figname=f"/home/figures/inference_precondition"
):
    for j, idx in enumerate(indices):
        plt.subplot(2, 5, 1 + j)
        tl = tlm_[idx, ...]
        gn = tl.T @ tl
        preconditioned_gn = preconditioners[idx, ...] @ gn
        plt.imshow(preconditioned_gn)
        plt.subplot(2, 5, 6 + j)
        _, sv, _ = np.linalg.svd(preconditioned_gn)
        plt.plot(sv)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def condition_numbers(
    preconditioners, tlm_, figname=f"/home/figures/condition_numbers"
):
    original_condition_number = np.linalg.cond((bt(tlm_) @ tlm_))
    sorted_indices = np.argsort(original_condition_number)
    prec_condition_number = np.linalg.cond((bt(tlm_) @ tlm_ @ preconditioners))
    plt.plot(original_condition_number[sorted_indices], ".")
    plt.plot(prec_condition_number[sorted_indices], ".")
    plt.yscale("log")
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

    logged_model = f"runs:/{run_id}/smoke_model"
    # mlflow.pyfunc.get_model_dependencies(logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"{loaded_model=}")

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    x_, fo_, tlm_ = zip(*data)

    x_ = np.asarray(x_)
    # approximations, preconditioners = construct_matrices(x_)
    approximations, preconditioners = sumLMP_ML(x_)
    tlm_ = np.asarray(tlm_)
    # plt.figure(figsize=(12, 6))
    indices = np.random.randint(0, len(x_), size=5)

    range_singular_values(tlm_)

    gauss_newton_approximation_svd(
        x_, tlm_, indices, figname="/home/figures/inference_approx"
    )

    preconditioned_svd(
        preconditioners, tlm_, indices, figname=f"/home/figures/inference_precondition"
    )

    condition_numbers(preconditioners, tlm_, figname=f"/home/figures/condition_numbers")

    sanity_check(
        approximations, preconditioners, indices, figname=f"/home/figures/sanity_check"
    )
