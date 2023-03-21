import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

plt.style.use("seaborn-v0_8")
import argparse

from DA_PoC.common.observation_operator import (
    IdentityObservationOperator,
)  # RandomObservationOperator,; LinearObervationOperator,
from DA_PoC.dynamical_systems.lorenz_numerical_model import (
    LorenzWrapper,
    burn_model,
    create_lorenz_model_observation,
)
from DA_PoC.variational.incrementalCG import Incremental4DVarCG, pad_ragged


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
    vecs, logsvals = pred[:, :20, :], pred[:, -1, :]
    sv = np.exp(logsvals)
    inv_sv = np.exp(-logsvals) - 1
    qi = bqr(vecs)
    # H = construct_matrix_USUt(qi, eli)
    mats = bmm(qi, (sv[..., None] * bt(qi)))
    prec = bmm(qi, (inv_sv[..., None] * bt(qi))) + np.eye(20)
    return mats, prec


plt.set_cmap("magma")


rng = np.random.default_rng(seed=93)


def main(config_DA, general_config, loaded_model=None):
    n = general_config["data"]["dimension"]

    lorenz = LorenzWrapper(n)
    lorenz.lorenz_model.dt = 0.01
    assimilation_window = [0.05, 0.4, 0.6, 0.8]  # 6 hours, 48 hours, 72 hours, 96 hours
    F = 8
    assimilation_window_timesteps = [
        int(wind / lorenz.lorenz_model.dt) for wind in assimilation_window
    ]
    nobs = assimilation_window_timesteps[1]

    sigma_b_sq = (0.04 * F) ** 2 + (0.1 * np.abs(0 - F)) ** 2
    charac_length = 1.5
    background_correlation = lambda x, y: np.exp(-((x - y) ** 2) / charac_length**2)
    x, y = np.meshgrid(np.arange(n), np.arange(n))

    B = sigma_b_sq * background_correlation(x, y)
    B_half = np.linalg.cholesky(B)
    B_inv = np.linalg.inv(B)

    sigma_obs_sq = (0.04 * F) ** 2 + (0.1 * np.abs(0 - F)) ** 2
    R = sigma_obs_sq * np.eye(n)
    R_half = np.linalg.cholesky(R)

    lorenz.background_error_cov_inv = B_inv
    lorenz.background = np.zeros(n)

    lorenz = LorenzWrapper(n)
    x0_t = burn_model(n, 1000)
    lorenz.n_total_obs = nobs

    m = n * (nobs + 1)
    lorenz.set_observations(nobs)
    lorenz.H = lambda x: x

    identity_obs_operator = IdentityObservationOperator(m)
    l_model_randobs = create_lorenz_model_observation(
        lorenz, identity_obs_operator, test_consistency=False
    )

    def get_next_observations(
        x_init, lorenz=lorenz, modsigsq=0.5, obssigsq=3, nobs=nobs
    ):
        lorenz.n_total_obs = nobs
        n = lorenz.state_dimension
        truth = np.empty((n, nobs + 1))
        curr_state = x_init
        truth[:, 0] = curr_state
        for i in range(nobs):
            curr_state = lorenz.lorenz_model.integrate(0, curr_state, 1)[1][
                :, 1
            ] + modsigsq * np.random.normal(size=(n))
            truth[:, i + 1] = curr_state
        obs = truth + obssigsq * np.random.normal(size=(n, (nobs + 1)))
        x_t = truth[:, -1]
        return obs, x_t, truth

    obs, x_t, truth = get_next_observations(x0_t)

    n_cycle = config_DA["DA"]["n_cycle"]  # 3
    n_outer = config_DA["DA"]["n_outer"]  # 10
    n_inner = config_DA["DA"]["n_inner"]  # 100
    log_file = config_DA["DA"]["log_file"]
    with open(log_file, "w+") as f:
        f.truncate()
    if loaded_model is not None:

        def MLpreconditioner(x):
            approx, prec = construct_matrices(loaded_model, x.reshape(1, n))
            return prec.squeeze()

        DA_ML = Incremental4DVarCG(
            state_dimension=n,
            bounds=None,
            numerical_model=l_model_randobs,
            observation_operator=identity_obs_operator,
            x0_t=np.random.normal(size=n),
            get_next_observations=get_next_observations,
            n_cycle=n_cycle,
            n_outer=n_outer,
            n_inner=n_inner,
            prec={"prec_name": MLpreconditioner, "side": "left"},
            plot=False,
            log_append=True,
        )
        DA_ML.GNlog_file = log_file
        DA_ML.exp_name = "ML"
        DA_ML.run()
    l_model_randobs.r = general_config["architecture"]["rank"]
    DA_precond = Incremental4DVarCG(
        state_dimension=n,
        bounds=None,
        numerical_model=l_model_randobs,
        observation_operator=identity_obs_operator,
        x0_t=np.random.normal(size=n),
        get_next_observations=get_next_observations,
        n_cycle=n_cycle,
        n_outer=n_outer,
        n_inner=n_inner,
        prec="spectralLMP",
        plot=False,
        log_append=True,
    )
    DA_precond.GNlog_file = log_file
    DA_precond.exp_name = "spectralLMP"
    DA_precond.run()

    DA_vanilla = Incremental4DVarCG(
        state_dimension=n,
        bounds=None,
        numerical_model=l_model_randobs,
        observation_operator=identity_obs_operator,
        x0_t=np.random.normal(size=n),
        get_next_observations=get_next_observations,
        n_cycle=n_cycle,
        n_outer=n_outer,
        n_inner=n_inner,
        prec=None,
        plot=False,
        log_append=True,
    )
    DA_vanilla.GNlog_file = log_file
    DA_vanilla.exp_name = "baseline"
    DA_vanilla.run()

    DA_vanilla.plot_residuals_inner_loop("C0", label="baseline", cumulative=False)
    DA_precond.plot_residuals_inner_loop("C1", label="spectral LMP", cumulative=False)
    if loaded_model is not None:
        DA_ML.plot_residuals_inner_loop("C2", label="ML", cumulative=False)
    plt.legend()
    plt.savefig("/home/figures/res_inner_loop.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use in inference")
    parser.add_argument("--config-DA", type=str, default="")
    parser.add_argument("--run-id-yaml", type=str, default="")
    parser.add_argument("--general-settings", type=str, default="")

    args = parser.parse_args()
    import yaml
    import mlflow
    import sys

    sys.path.append("..")
    # try:
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
    config_DA = OmegaConf.load(args.config_DA)
    config = OmegaConf.load(args.general_settings)

    main(config_DA, config, loaded_model)
