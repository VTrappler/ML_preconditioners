import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

plt.style.use("seaborn-v0_8")
import argparse
import logging
import os

import seaborn as sns
from DA_PoC.common.observation_operator import (
    IdentityObservationOperator,
)  # RandomObservationOperator,; LinearObervationOperator,
from DA_PoC.dynamical_systems.lorenz_numerical_model import (
    LorenzWrapper,
    burn_model,
    create_lorenz_model_observation,
)
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


def construct_matrices(loaded_model, x_):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    vecs, logsvals = pred[:, :-1, :], pred[:, -1, :]
    sv = np.exp(logsvals)
    # logging.info(f"MLsvals={np.exp(logsvals)}")
    inv_sv = np.exp(-logsvals) - 1
    qi = bqr(vecs)
    # H = construct_matrix_USUt(qi, eli)
    mats = bmm(qi, (sv[..., None] * bt(qi)))
    prec = bmm(qi, (inv_sv[..., None] * bt(qi))) + np.eye(vecs.shape[1])
    logging.info(f"svd approximation= {np.linalg.svd(mats)[1]}")
    # logging.info(f"svd preconditioner= {np.linalg.svd(prec)[1]}")

    return mats, prec


def construct_LMP_naive(U, A, shift=1):
    identity = np.eye(A.shape[0])
    normalizing_matrix = np.linalg.inv(U.T @ A @ U)
    AU = A @ U
    Hk = (identity - U @ normalizing_matrix @ AU.T) @ (
        identity - AU @ normalizing_matrix @ U.T
    ) + shift * U @ normalizing_matrix @ U.T
    return Hk


def construct_LMP_direct(U, AU):
    identity = np.eye(U.shape[0])
    normalizing_matrix = np.linalg.inv(U.T @ AU)
    Hk = (identity - U @ normalizing_matrix @ AU.T) @ (
        identity - AU @ normalizing_matrix @ U.T
    ) + U @ normalizing_matrix @ U.T
    return Hk


def construct_LMP_ML(loaded_model, x_):
    prediction = loaded_model.predict(np.asarray(x_).astype("f"))
    S, AS = prediction[..., 0], prediction[..., 1]
    return construct_LMP_direct(S.squeeze(), AS.squeeze())


plt.set_cmap("magma")


rng = np.random.default_rng(seed=93)


def main(config, loaded_model=None):
    n = config["model"]["dimension"]

    # lorenz = LorenzWrapper(n)
    # lorenz.lorenz_model.dt = 0.01
    # assimilation_window = [0.05, 0.4, 0.6, 0.8]  # 6 hours, 48 hours, 72 hours, 96 hours
    # F = 8
    # assimilation_window_timesteps = [
    #     int(wind / lorenz.lorenz_model.dt) for wind in assimilation_window
    # ]
    # nobs = assimilation_window_timesteps[1]

    # sigma_b_sq = (0.04 * F) ** 2 + (0.1 * np.abs(0 - F)) ** 2
    # charac_length = 1.5
    # background_correlation = lambda x, y: np.exp(-((x - y) ** 2) / charac_length**2)
    # x, y = np.meshgrid(np.arange(n), np.arange(n))

    # B = sigma_b_sq * background_correlation(x, y)
    # B_half = np.linalg.cholesky(B)
    # B_inv = np.linalg.inv(B)

    # sigma_obs_sq = (0.04 * F) ** 2 + (0.1 * np.abs(0 - F)) ** 2
    # R = sigma_obs_sq * np.eye(n)
    # R_half = np.linalg.cholesky(R)

    # lorenz.background_error_cov_inv = B_inv
    # lorenz.background = np.zeros(n)

    window = 10
    lorenz = LorenzWrapper(n)
    x0_t = burn_model(n, 1000)
    lorenz.n_total_obs = window

    m = n * (window + 1)
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
            x0_analysis=x0_t,
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

    # Diagnostic SVD

    # if prec_type == "general":
    #     b = -self.gradient(x)
    #     args = {
    #         "A": GtG,
    #         "b": b,
    #         "x": x,
    #     }
    #     A_to_inv, b_to_inv = prec(**args)
    #     return solve_cg(A_to_inv, b_to_inv, maxiter=iter_inner)

    def MLpreconditioner_LMP(x):
        Hk = construct_LMP_ML(loaded_model, x.reshape(1, n))
        return Hk.squeeze()

    _ = create_DA_experiment(
        "ML_LMP", prec={"prec_name": MLpreconditioner_LMP, "prec_type": "left"}
    )

    def MLpreconditioner_LMP_A(A, b, x):
        prediction = loaded_model.predict(np.asarray(x).reshape(1, n).astype("f"))
        S = prediction[..., 0].squeeze()
        Hk = construct_LMP_direct(S, A @ S)
        return Hk @ A, Hk @ b

    _ = create_DA_experiment(
        "ML_naive", prec={"prec_name": MLpreconditioner_LMP_A, "prec_type": "general"}
    )

    # _ = create_DA_experiment(
    #     "deflation_ML",
    #     prec={
    #         "prec_name": lambda x: MLpreconditioner_svd(x, qr=False),
    #         "prec_type": "deflation",
    #     },
    # )

    l_model_randobs.r = config["architecture"]["rank"]
    DA_spectralLMP = create_DA_experiment("spectralLMP", prec="spectralLMP")

    # BASELINE --------------------------------------------------------------------
    DA_baseline = create_DA_experiment("baseline", prec=None)

    #
    #
    #
    #
    # Run experiments

    for exp_name, DA_exp in DA_exp_dict.items():
        print(f"\n--- {exp_name} ---\n")
        DA_exp.run()
    dfs = []
    for i, (exp_name, DA_exp) in enumerate(DA_exp_dict.items()):
        DA_exp.plot_residuals_inner_loop(
            f"C{i}", label=DA_exp.exp_name, cumulative=False
        )
        cond, niter = DA_exp.extract_condition_niter()
        dfs.append(
            pd.DataFrame(
                {"exp_name": DA_exp.exp_name, "condition": cond, "niter": niter}
            )
        )
    dataframe = pd.concat(dfs)

    plt.legend()
    plt.ylim([1e-9, 1e2])
    plt.savefig(os.path.join(artifacts_path, "res_inner_loop.png"))
    plt.close()

    plt.subplot(1, 2, 1)
    sns.boxplot(dataframe, x="exp_name", y="condition", orient="v")
    plt.ylabel("Condition")
    plt.yscale("log")
    plt.subplot(1, 2, 2)
    sns.boxplot(dataframe, x="exp_name", y="niter", orient="v")
    plt.ylabel("# iter")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_path, "cond_niter_boxplot.png"))
    plt.close()


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
