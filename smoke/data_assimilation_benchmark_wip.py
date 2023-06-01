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
import sys

sys.path.append("..")
# import ..prec_models as prec_models
from DA_PoC.common.linearsolver import (
    construct_LMP,
    solve_cg,
    conjGrad,
)

from DA_PoC.common.preconditioned_solvers import PreconditionedSolver, SplitPrec

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
    return Sr.squeeze(), Ur.squeeze()


class MLPrec(PreconditionedSolver):
    def __init__(self, ML_model, tol: float = 1e-8, maxiter: int = 40):
        super().__init__(tol, maxiter)
        self.ML_model = ML_model

    def power_prec(self, x_, power):
        Sr, Ur = construct_svd_ML(self.ML_model, x_[None, ...], qr=True)
        return Ur @ ((Sr[None, :] ** (power) - 1) * Ur).T + np.eye(Ur.shape[0])

    def low_rank_reconstruction(self, x_, power):
        Sr, Ur = construct_svd_ML(self.ML_model, x_[None, ...], qr=True)
        return Ur @ ((Sr[None, :] ** (power)) * Ur).T

    def __call__(self, A, b, x, maxiter):
        raise NotImplementedError()


class MLSplitPrec(MLPrec):
    def __init__(self, ML_model, tol: float = 1e-8, maxiter: int = 40):
        super().__init__(ML_model, tol, maxiter)

    def __call__(self, A, b, x, maxiter):
        L = self.power_prec(x, power=-0.5)
        # Linv = self.power_prec(A, power=0.5)
        x_hat, res_dict = conjGrad(
            L.T @ A @ L,
            0 * b,
            L.T @ b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return L @ x_hat, res_dict


class MLSplitPrecTruncated(MLSplitPrec):
    def __init__(self, ML_model, rank: int, tol: float = 1e-8, maxiter: int = 40):
        super().__init__(ML_model, tol, maxiter)
        self.rank = rank

    def power_prec(self, x_, power):
        Sr, Ur = construct_svd_ML(self.ML_model, x_[None, ...], qr=True)
        sorted_idx = np.argsort(Sr)[::-1]
        Sr_s, Ur_s = Sr[sorted_idx], Ur[:, sorted_idx]
        Sr = Sr_s[: self.rank]
        Ur = Ur_s[:, : self.rank]
        return Ur @ ((Sr[None, :] ** (power) - 1) * Ur).T + np.eye(Ur.shape[0])

    def __call__(self, A, b, x, maxiter):
        L = self.power_prec(x, power=-0.5)
        # Linv = self.power_prec(A, power=0.5)
        x_hat, res_dict = conjGrad(
            L.T @ A @ L,
            0 * b,
            L.T @ b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return L @ x_hat, res_dict


class MLNaiveLMP(MLSplitPrecTruncated):
    def __init__(self, ML_model, rank: int, tol: float = 1e-8, maxiter: int = 40):
        super().__init__(ML_model, rank, tol, maxiter)

    def __call__(self, A, b, x, maxiter):
        Sr, Ur = construct_svd_ML(self.ML_model, x[None, ...], qr=True)
        sorted_idx = np.argsort(Sr)[::-1]
        Sr_s, Ur_s = Sr[sorted_idx], Ur[:, sorted_idx]
        Sr = Sr_s[: self.rank]
        Ur = Ur_s[:, : self.rank]
        H = construct_LMP(40, Ur, A @ Ur)
        x_hat, res_dict = conjGrad(
            H @ A,
            0 * b,
            H @ b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return x_hat, res_dict


def naive_LMP(A, b, x, loaded_model):
    pred = loaded_model.predict(np.asarray(x).reshape(1, 40).astype("f"))
    Ur = pred[:, :-1, :].squeeze()
    H = construct_LMP(40, Ur, A @ Ur)
    return H @ A, H @ b


class MLPseudoInverse(MLPrec):
    def __init__(self, ML_model, tol: float = 1e-8, maxiter: int = 40):
        super().__init__(ML_model, tol, maxiter)

    def __call__(self, A, b, x, maxiter):
        pseudo_inv = self.low_rank_reconstruction(x, power=-1)
        x_hat, res_dict = conjGrad(
            A,
            pseudo_inv @ b,
            b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return x_hat, res_dict


class MLPseudoInverse2(MLPrec):
    def __init__(self, ML_model, tol: float = 1e-8, maxiter: int = 40):
        super().__init__(ML_model, tol, maxiter)

    def __call__(self, A, b, x, maxiter):
        pseudo_inv = self.power_prec(x, power=-1)
        x_hat, res_dict = conjGrad(
            A,
            pseudo_inv @ b,
            b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return x_hat, res_dict


class SplitPrec(PreconditionedSolver):
    def __init__(
        self, num_model, rank: int = None, tol: float = 1e-8, maxiter: int = 40
    ):
        super().__init__(tol, maxiter)
        self.num_model = num_model
        if rank is None:
            self.rank = self.num_model.r
        else:
            self.rank = rank

    def power_prec(self, x_, power):
        GtG = self.num_model.gauss_newton_hessian_matrix(x_)
        U, S, _ = np.linalg.svd(GtG)
        Ur, Sr = U[:, : self.rank], S[: self.rank]
        # Ur += 0.1 * np.random.normal(size=Ur.shape)
        # Ur = np.linalg.qr(Ur)[0]
        return Ur @ ((Sr[None, :] ** (power) - 1) * Ur).T + np.eye(Ur.shape[0])

    def __call__(self, A, b, x, maxiter):
        L = self.power_prec(x, power=-0.5)
        # Linv = self.power_prec(A, power=0.5)
        x_hat, res_dict = conjGrad(
            L.T @ A @ L,
            0 * b,
            L.T @ b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return L @ x_hat, res_dict


class PseudoInvExact(SplitPrec):
    def low_rank_reconstruction(self, x_, power):
        GtG = self.num_model.gauss_newton_hessian_matrix(x_)
        U, S, _ = np.linalg.svd(GtG)
        Ur, Sr = U[:, : self.rank], S[: self.rank]
        # Ur += 0.1 * np.random.normal(size=Ur.shape)
        return Ur @ ((Sr[None, :] ** (power)) * Ur).T

    def __call__(self, A, b, x, maxiter):
        # L = self.power_prec(x, power=-0.5)
        pinv = self.low_rank_reconstruction(x, power=-1)
        x_hat, res_dict = conjGrad(
            A,
            pinv @ b,
            b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return x_hat, res_dict


def construct_projector_exact(num_model, x_, rk):
    GtG = num_model.gauss_newton_hessian_matrix(x_)
    U, S, _ = np.linalg.svd(GtG)
    return S[:rk], U[:, :rk]


def preconditioner_from_SVD(S, U, regul=0.0):
    regul_sv = (S / (regul + S**2)) - 1
    return U @ (regul_sv * U).T + np.eye(40)


def naive_LMP(A, b, x, loaded_model):
    pred = loaded_model.predict(np.asarray(x).reshape(1, 40).astype("f"))
    Ur = pred[:, :-1, :].squeeze()
    H = construct_LMP(40, Ur, A @ Ur)
    return H @ A, H @ b


def regularized_balance(alpha_regul):
    def ML_balance(x):
        pred = loaded_model.predict(np.asarray(x).reshape(1, 40).astype("f"))
        Ur, logsvals = pred[:, :-1, :], pred[:, -1, :]
        Sr = np.exp(logsvals).squeeze()
        Ur = bqr(Ur).squeeze()
        return preconditioner_from_SVD(Sr, Ur, regul=alpha_regul)

    return ML_balance


rng = np.random.default_rng(seed=93)


def main(config, loaded_model=None):
    n = config["model"]["dimension"]

    window = config["model"]["window"]
    lorenz = LorenzWrapper(n)
    x0_t = burn_model(lorenz, 1000)
    lorenz.n_total_obs = window

    m = n * (window + 1)

    # lorenz.background_error_cov_inv = np.eye(n)
    # lorenz.background = np.zeros(n)

    lorenz.set_observations(window)
    lorenz.H = lambda x: x

    identity_obs_operator = IdentityObservationOperator(m)
    l_model_randobs = create_lorenz_model_observation(
        lorenz, identity_obs_operator, test_consistency=False
    )
    l_model_randobs.r = config["architecture"]["rank"]

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

    # _ = create_DA_experiment("Split", prec=SplitPrec(l_model_randobs, maxiter=n_inner))

    # _ = create_DA_experiment("MLsplit", prec=MLSplitPrec(loaded_model, maxiter=n_inner))

    DA_baseline = create_DA_experiment("baseline", prec=None)

    for i in [
        40,
        35,
        30,
        25,
        20,
        15,
    ]:
        _ = create_DA_experiment(
            f"MLSplit_{i}",
            prec=MLSplitPrecTruncated(loaded_model, rank=i, maxiter=n_inner),
        )
        _ = create_DA_experiment(
            f"MLNaiveLMP_{i}",
            prec=MLNaiveLMP(loaded_model, rank=i, maxiter=n_inner),
        )
        _ = create_DA_experiment(
            f"Split_{i}", prec=SplitPrec(l_model_randobs, rank=i, maxiter=n_inner)
        )

    #     )
    # _ = create_DA_experiment(
    #     "PseudoInv", prec=MLPseudoInverse(loaded_model, maxiter=n_inner)
    # )

    # _ = create_DA_experiment(
    #     "PseudoInv2", prec=MLPseudoInverse2(loaded_model, maxiter=n_inner)
    # )

    # DA_spectralLMP = create_DA_experiment("spectralLMP", prec="spectralLMP")

    # def prec_naive_ML_LMP(A, b, x):
    #     return naive_LMP(A, b, x, loaded_model)

    # _ = create_DA_experiment(
    #     "naiveML_LMP", prec={"prec_type": "general", "prec_name": prec_naive_ML_LMP}
    # )

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
    df_baseline = df[df["name"] == "baseline"]
    mean_cond_baseline = df_baseline["cond"].mean()
    mean_niter_baseline = df_baseline["niter"].mean()

    plt.subplot(1, 2, 1)
    ax = sns.boxplot(df, x="name", y="cond", orient="v")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
    plt.ylabel("Condition number")
    plt.axhline(y=mean_cond_baseline, color="r")

    plt.subplot(1, 2, 2)
    ax = sns.boxplot(df, x="name", y="niter", orient="v")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
    plt.ylabel("CG iterations")
    plt.axhline(y=mean_niter_baseline, color="r")

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_path, "cond_niter_boxplot.png"))
    plt.close()

    plt.subplot(1, 2, 1)
    ax = sns.violinplot(df, x="name", y="cond", orient="v")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
    plt.ylabel("Condition number")
    plt.axhline(y=mean_cond_baseline, color="r")

    plt.subplot(1, 2, 2)
    ax = sns.violinplot(df, x="name", y="niter", orient="v")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
    plt.ylabel("CG iterations")
    plt.axhline(y=mean_niter_baseline, color="r")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_path, "cond_niter_violin.png"))


if __name__ == "__main__":
    print("WIP")
    parser = argparse.ArgumentParser(description="Use in inference")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--run-id-yaml", type=str, default="")

    args = parser.parse_args()
    import mlflow
    import yaml

    sys.path.append("..")
    with open(args.run_id_yaml, "r") as fstream:
        run_id_yaml = yaml.safe_load(fstream)
        run_id = run_id_yaml["run_id"]
    logged_model = f"runs:/{run_id}/smoke_model"
    mlflow.pyfunc.get_model_dependencies(logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"{loaded_model=}")
    # except:
    # loaded_model = None
    config = OmegaConf.load(args.config)
    main(config, loaded_model)
