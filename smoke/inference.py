import mlflow
import numpy as np
import os
import sys
import argparse

sys.path.append("/home/")


def bmm(a, b):
    return np.einsum("Bij,Bjk ->Bik", a, b)


def bt(a):
    return np.einsum("Bij->Bji", a)


def bqr(a):
    Q = np.empty_like(a)
    for i, mat in enumerate(a):
        Q[i, ...] = np.linalg.qr(mat)[0]
    return Q


def construct_matrices(x_):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    vecs, logsvals = pred[:, :20, :], pred[:, -1, :]
    sv = np.exp(logsvals)
    inv_sv = np.exp(-logsvals)
    qi = bqr(vecs)
    # H = construct_matrix_USUt(qi, eli)
    mats = bmm(qi, (sv[..., None] * bt(qi)))
    prec = bmm(qi, (inv_sv[..., None] * bt(qi)))
    return mats, prec


def get_approximate_singular_val(x_):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    logsvals = pred[:, -1, :]
    sv = np.exp(logsvals)
    return sv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Surrogate in inference")
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()
    print(os.getenv("MLFLOW_TRACKING_URI"))
    logged_model = f"runs:/{args.run_id}/smoke_model"
    # mlflow.pyfunc.get_model_dependencies(logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"{loaded_model=}")

    import pickle

    with open("/root/raw_data/data.pkl", "rb") as handle:
        data = pickle.load(handle)
    x_, fo_, tlm_ = zip(*data)

    import matplotlib.pyplot as plt

    x_ = np.asarray(x_)
    approximations, preconditioners = construct_matrices(x_)
    tlm_ = np.asarray(tlm_)
    # plt.figure(figsize=(12, 6))
    for j, idx in enumerate(np.random.randint(0, 10000, size=5)):
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
            get_approximate_singular_val(x_[idx].reshape(-1, 20)).flatten()
        )[::-1]
        plt.plot(approx_sv, color="red")
    plt.tight_layout()
    plt.savefig(f"/home/figures/inference_approx_{args.run_id}")
    plt.close()

    for j, idx in enumerate(np.random.randint(0, 10000, size=5)):
        plt.subplot(2, 5, 1 + j)
        tl = tlm_[idx, ...]
        gn = tl.T @ tl
        preconditioned_gn = gn @ preconditioners[idx, ...]
        plt.imshow(preconditioned_gn)
        plt.subplot(2, 5, 6 + j)
        _, sv, _ = np.linalg.svd(preconditioned_gn)
        plt.plot(sv)

    plt.tight_layout()
    plt.savefig(f"/home/figures/inference_precondition_{args.run_id}")
    plt.close()

    original_condition_number = np.linalg.cond((bt(tlm_) @ tlm_))
    sorted_indices = np.argsort(original_condition_number)
    prec_condition_number = np.linalg.cond((bt(tlm_) @ tlm_ @ preconditioners))
    plt.plot(original_condition_number[sorted_indices], ".")
    plt.plot(prec_condition_number[sorted_indices], ".")
    plt.yscale("log")
    plt.savefig(f"/home/figures/condition_numbers_{args.run_id}")
    plt.close()
