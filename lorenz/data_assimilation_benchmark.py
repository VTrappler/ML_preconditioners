import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import seaborn as sns

plt.style.use("seaborn-v0_8")
plt.set_cmap("magma")

import argparse
import logging
import os
import pandas as pd
from DA_PoC.common.observation_operator import (
    IdentityObservationOperator,
)  # RandomObservationOperator,; LinearObervationOperator,
from DA_PoC.dynamical_systems.lorenz_numerical_model import (
    LorenzWrapper,
    burn_model,
    create_lorenz_model_observation,
)

from DA_PoC.common.linearsolver import construct_LMP, solve_cg

from DA_PoC.variational.incrementalCG import Incremental4DVarCG, pad_ragged

logging.basicConfig(level=logging.INFO)
logs_path = os.path.join(os.sep, "root", "log_dump", "smoke")
smoke_path = os.path.join(os.sep, "home", "smoke")
artifacts_path = os.path.join(smoke_path, "artifacts")


def bmm(a, b):
    return np.einsum("Bij,Bjk ->Bik", a, b)


def bt(a):
    return np.einsum("Bij->Bji", a)


def bqr(a):
    Q = np.empty_like(a)
    for i, mat in enumerate(a):
        Q[i, ...] = np.linalg.qr(mat)[0]
    return Q


def construct_matrices(loaded_model, x_, alpha_regul=0):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    vecs, logsvals = pred[:, :-1, :], pred[:, -1, :]
    sv = np.exp(logsvals)
    # logging.info(f"MLsvals={np.exp(logsvals)}")
    regul_sv = sv / (alpha_regul + sv**2) - 1
    qi = bqr(vecs)
    # H = construct_matrix_USUt(qi, eli)
    mats = bmm(qi, (sv[..., None] * bt(qi)))
    prec = bmm(qi, (regul_sv[..., None] * bt(qi))) + np.eye(vecs.shape[1])
    logging.info(f"svd approximation= {np.linalg.svd(mats)[1]}")
    # logging.info(f"svd preconditioner= {np.linalg.svd(prec)[1]}")

    return mats, prec


def construct_svd_ML(loaded_model, x_, qr=False):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    Ur, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    if qr:
        Ur = bqr(Ur)
    # Ur = bqr(vecs)
    # proj = bmm(qi, (inv_sv[..., None] * bt(qi)))
    return Sr.squeeze(), Ur.squeeze()


def construct_projector_exact(num_model, x_, rk):
    GtG = num_model.gauss_newton_hessian_matrix(x_)
    U, S, _ = np.linalg.svd(GtG)
    return S[:rk], U[:, :rk]


def sumLMP_exact(num_model, x_, rk):
    Sr, Ur = construct_projector_exact(num_model, x_, rk)
    n = len(x_)
    acc = np.zeros((n, n))
    for i in range(rk):
        acc += (1 - (1 / Sr[i])) * Ur[:, i].reshape(n, 1) @ Ur[:, i].reshape(1, n)
    return np.eye(n) - acc


def sumLMP_ML(loaded_model, x_, qr=True):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    Ur, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    if qr:
        Ur = bqr(Ur)
    n = Ur.shape[1]
    r = Ur.shape[-1]
    acc = np.zeros((len(pred), n, n))
    for i in range(r):
        uu = Ur[..., i].reshape((len(pred), n, 1))
        uuprime = bmm(uu, bt(uu))
        acc += ((1 - Sr[:, i] ** (-1))).reshape(len(pred), 1, 1) * uuprime
    prec = np.eye(n).reshape(1, n, n) - acc
    mats = bmm(Ur, (Sr[..., None] * bt(Ur)))
    return mats, prec


def preconditioner_from_SVD(S, U, regul=0.0):
    regul_sv = (S / (regul + S**2)) - 1
    return U @ (regul_sv * U).T + np.eye(100)


def naive_LMP(A, b, x, loaded_model):
    pred = loaded_model.predict(np.asarray(x).reshape(1, 100).astype("f"))
    Ur = pred[:, :-1, :].squeeze()
    H = construct_LMP(100, Ur, A @ Ur)
    return H @ A, H @ b


def regularized_balance(alpha_regul):
    def ML_balance(x):
        pred = loaded_model.predict(np.asarray(x).reshape(1, 100).astype("f"))
        Ur, logsvals = pred[:, :-1, :], pred[:, -1, :]
        Sr = np.exp(logsvals).squeeze()
        Ur = bqr(Ur).squeeze()
        return preconditioner_from_SVD(Sr, Ur, regul=alpha_regul)

    return ML_balance


rng = np.random.default_rng(seed=93)


def main(config, loaded_model=None):
    n = config["model"]["dimension"]

    window = 10
    lorenz = LorenzWrapper(n)
    x0_t = burn_model(lorenz, 1000)
    lorenz.n_total_obs = window

    m = n * (window + 1)

    lorenz.background_error_cov_inv = np.eye(n)
    lorenz.background = np.zeros(n)

    lorenz.set_observations(window)
    lorenz.H = lambda x: x

    identity_obs_operator = IdentityObservationOperator(m)
    l_model_randobs = create_lorenz_model_observation(
        lorenz, identity_obs_operator, test_consistency=False
    )

    def get_next_obs(x0):
        return lorenz.get_next_observations(
            x0,
            model_error_sqrt=config["DA"]["model_error_sqrt"],
            obs_error_sqrt=config["DA"]["obs_error_sqrt"],
        )

    # obs, x_t, truth = get_next_obs(x0_t)

    n_cycle = config["DA"]["n_cycle"]  # 3
    n_outer = config["DA"]["n_outer"]  # 10
    n_inner = config["DA"]["n_inner"]  # 100
    log_file = config["DA"]["log_file"]

    with open(log_file, "w+") as f:
        f.truncate()

    DA_exp_dict = {}

    def create_DA_experiment(exp_name, prec):
        DA_exp = Incremental4DVarCG(
            state_dimension=n,
            bounds=None,
            numerical_model=l_model_randobs,
            observation_operator=identity_obs_operator,
            x0_run=x0_t,
            x0_analysis=None,
            get_next_observations=get_next_obs,
            n_cycle=n_cycle,
            n_outer=n_outer,
            n_inner=n_inner,
            prec=prec,
            plot=False,
            log_append=True,
            save_all=True,
        )
        DA_exp.GNlog_file = log_file
        DA_exp.exp_name = exp_name
        DA_exp_dict[exp_name] = DA_exp
        return DA_exp

    # def MLpreconditioner_LMP(x):
    #     _, prec = sumLMP_ML(loaded_model, x.reshape(1, n), qr=True)
    #     return prec.squeeze()

    # _ = create_DA_experiment(
    #     "ML_LMP", prec={"prec_name": MLpreconditioner_LMP, "prec_type": "right"}
    # )

    # def MLpreconditioner_svd(x, qr=False):
    #     return construct_svd_ML(loaded_model, x.reshape(1, n), qr)

    # DA_diag = create_DA_experiment(
    #     "diagnostic",
    #     prec={"prec_name": MLpreconditioner_svd, "prec_type": "svd_diagnostic"},
    # )

    # def deflation_preconditioner(x):
    #     return construct_projector_exact(
    #         l_model_randobs, x, config["architecture"]["rank"]
    #     )

    # DA_deflation = create_DA_experiment(
    #     "deflation_ML",
    #     prec={
    #         "prec_name": lambda x: MLpreconditioner_svd(x, qr=True),
    #         "prec_type": "deflation",
    #     },
    # )
    def prec_naive_ML_LMP(A, b, x):
        return naive_LMP(A, b, x, loaded_model)

    _ = create_DA_experiment(
        "naiveML_LMP", prec={"prec_type": "general", "prec_name": prec_naive_ML_LMP}
    )

    for regul in [0.0, 1.0, 2.0, 10.0]:
        _ = create_DA_experiment(
            f"regul_{int(regul)}",
            # f"ML",
            prec={
                "prec_name": regularized_balance(regul),
                "prec_type": "right",
            },
        )

    # def sumLMP_preconditioner(x):
    #     return sumLMP_exact(l_model_randobs, x, config["architecture"]["rank"])

    # DA_sumLMP = create_DA_experiment(
    #     "sumLMP", {"prec_name": sumLMP_preconditioner, "prec_type": "right"}
    # )
    l_model_randobs.r = config["architecture"]["rank"]
    DA_spectralLMP = create_DA_experiment("spectralLMP", prec="spectralLMP")
    DA_baseline = create_DA_experiment("baseline", prec=None)

    df_list = []

    for exp_name, DA_exp in DA_exp_dict.items():
        print(f"\n--- {exp_name} ---\n")
        DA_exp.run()
        cond, niter = DA_exp.extract_condition_niter()
        df_list.append(pd.DataFrame({"niter": niter, "cond": cond, "name": exp_name}))

    for i, (exp_name, DA_exp) in enumerate(DA_exp_dict.items()):
        DA_exp.plot_residuals_inner_loop(
            f"C{i}", label=DA_exp.exp_name, cumulative=False
        )
    plt.legend()
    plt.ylim([1e-9, 1e2])
    plt.savefig(os.path.join(artifacts_path, "res_inner_loop.png"))
    plt.close()

    df = pd.concat(df_list)
    plt.subplot(1, 2, 1)
    ax = sns.boxplot(df, x="name", y="cond", orient="v")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
    plt.ylabel("Condition")
    plt.subplot(1, 2, 2)
    ax = sns.boxplot(df, x="name", y="niter", orient="v")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
    plt.ylabel("# iter")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_path, "cond_niter_boxplot.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use in inference")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--run-id-yaml", type=str, default="")

    args = parser.parse_args()
    import sys

    import mlflow
    import yaml

    sys.path.append("..")
    with open(args.run_id_yaml, "r") as fstream:
        run_id_yaml = yaml.safe_load(fstream)
        run_id = run_id_yaml["run_id"]
    print(run_id)
    logged_model = f"runs:/{run_id}/smoke_model"
    # mlflow.pyfunc.get_model_dependencies(logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"{loaded_model=}")
    # except:
    # loaded_model = None
    config = OmegaConf.load(args.config)
    main(config, loaded_model)
