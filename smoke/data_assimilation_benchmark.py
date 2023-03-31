import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

plt.style.use("seaborn-v0_8")
import argparse
import logging
import os

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
    logging.info(f"svd preconditioner= {np.linalg.svd(prec)[1]}")

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


def sumLMP_ML(loaded_model, x_):
    pred = loaded_model.predict(np.asarray(x_).astype("f"))
    vecs, logsvals = pred[:, :-1, :], pred[:, -1, :]
    Sr = np.exp(logsvals)
    Ur = bqr(vecs)
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

    obs, x_t, truth = get_next_obs(x0_t)

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
            x0_t=np.random.normal(size=n),
            get_next_observations=get_next_obs,
            n_cycle=n_cycle,
            n_outer=n_outer,
            n_inner=n_inner,
            prec=prec,
            plot=False,
            log_append=True,
        )
        DA_exp.GNlog_file = log_file
        DA_exp.exp_name = exp_name
        DA_exp_dict[exp_name] = DA_exp
        return DA_exp

    if loaded_model is not None:

        def MLpreconditioner(x):
            approx, prec = sumLMP_ML(loaded_model, x.reshape(1, n))
            return prec.squeeze()

        def MLpreconditioner_svd(x, qr=False):
            return construct_svd_ML(loaded_model, x.reshape(1, n), qr)

        DA_diag = create_DA_experiment(
            "diagnostic",
            prec={"prec_name": MLpreconditioner_svd, "prec_type": "svd_diagnostic"},
        )

        create_DA_experiment(
            "ML_LMP", prec={"prec_name": MLpreconditioner, "prec_type": "right"}
        )

    def sumLMP_preconditioner(x):
        return sumLMP_exact(l_model_randobs, x, config["architecture"]["rank"])

    # DA_sumLMP = create_DA_experiment(
    #     "sumLMP", {"prec_name": sumLMP_preconditioner, "prec_type": "left"}
    # )

    def deflation_preconditioner(x):
        return construct_projector_exact(
            l_model_randobs, x, config["architecture"]["rank"]
        )

    DA_deflation = create_DA_experiment(
        "deflation_ML",
        prec={
            "prec_name": lambda x: MLpreconditioner_svd(x, qr=True),
            "prec_type": "deflation",
        },
    )
    l_model_randobs.r = config["architecture"]["rank"]
    # DA_spectralLMP = create_DA_experiment("spectralLMP", prec="spectralLMP")
    DA_baseline = create_DA_experiment("baseline", prec=None)

    for exp_name, DA_exp in DA_exp_dict.items():
        print(f"\n--- {exp_name} ---\n")
        DA_exp.run()

    for i, (exp_name, DA_exp) in enumerate(DA_exp_dict.items()):
        DA_exp.plot_residuals_inner_loop(
            f"C{i}", label=DA_exp.exp_name, cumulative=False
        )
    plt.legend()
    plt.savefig(os.path.join(artifacts_path, "res_inner_loop.png"))
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
