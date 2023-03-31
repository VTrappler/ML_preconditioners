import argparse
import pickle

import numpy as np
from DA_PoC.dynamical_systems.lorenz_numerical_model import (
    LorenzWrapper,
    burn_model,
    create_lorenz_model_observation,
)

from DA_PoC.common.observation_operator import (
    IdentityObservationOperator,
)

from omegaconf import OmegaConf
from tqdm.rich import tqdm


def generate_datapoint(x_current, lorenz, time_window):
    n = lorenz.state_dimension
    history = lorenz.lorenz_model.integrate(0, x_current, time_window)
    tlm = lorenz.lorenz_model.construct_tlm_matrix(0, x_current, time_window)
    tlm = tlm.reshape(-1, n)
    return (x_current, history[1], tlm), (history[1][:, -1])


# def generate_datapoint_GNmatrix(x_current, lorenz, time_window):
#     n = lorenz.state_dimension
#     history = lorenz.lorenz_model.integrate(0, x_current, time_window)
#     return (x_current, history[1], tlm), (history[1][:, -1])


def generate_dataset(lorenz, x0, assimilation_window, nsamples):
    train = []
    x = x0
    for _ in tqdm.trange(nsamples):
        datapoint, x = generate_datapoint(
            x_current=x, lorenz=lorenz, nobs=assimilation_window
        )
        train.append(datapoint)
    return train


def save_data(data, path):
    pickle.dump(data, open(f"{path}", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the training data for the Lorenz model"
    )
    parser.add_argument(
        "--config",
        help="configuration file *.yml",
        type=str,
        required=False,
        default="data_params.yml",
    )
    parser.add_argument("-dim", type=int, help="State dimension of the Lorenz model")
    parser.add_argument("-n_total_obs", type=int, help="Number of observations")
    parser.add_argument("-N", type=int, help="Number of training data to save")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    # def H_nl(x, gamma=4):
    #     return 2 * x + 1

    if len(args.config) > 0:
        conf = OmegaConf.load(args.config)
        dim = conf.model["dimension"]
        nsamples = conf.model["nsamples"]
        window = conf.model["window"]
        if "output" in conf.model.keys():
            target = conf.model["output"]
        else:
            target = f"raw_data/{dim}_{window}obs_{nsamples}.pkl"

    else:
        nsamples = args.N
        dim = args.dim
        window = args.n_total_obs
        target = args.target

    print(f"Training tuples to save: {args.N}")
    print(f"into {target}")

    lorenz = LorenzWrapper(dim)
    assimilation_window = [0.05, 0.4, 0.6, 0.8]  # 6 hours, 48 hours, 72 hours, 96 hours
    F = 8
    # assimilation_window_timesteps = [
    #     int(window / lorenz.lorenz_model.dt) for window in assimilation_window
    # ]
    # nobs = assimilation_window_timesteps[1]
    sigma_b_sq = (0.04 * F) ** 2 + (0.1 * np.abs(0 - F)) ** 2
    charac_length = 1.5
    background_correlation = lambda x, y: np.exp(-((x - y) ** 2) / charac_length**2)
    x, y = np.meshgrid(np.arange(dim), np.arange(dim))

    B = sigma_b_sq * background_correlation(x, y)
    B_half = np.linalg.cholesky(B)
    B_inv = np.linalg.inv(B)

    sigma_obs_sq = (0.04 * F) ** 2 + (0.1 * np.abs(0 - F)) ** 2
    R = sigma_obs_sq * np.eye(dim)
    R_half = np.linalg.cholesky(R)

    lorenz.background_error_cov_inv = B_inv
    lorenz.background = np.zeros(dim)

    lorenz.create_and_burn_truth()

    identity_obs_operator = IdentityObservationOperator(m)
    # plt.imshow(random_obs_operator.H)
    # m = n * (nobs + 1)
    lorenz.H = lambda x: x
    num_model = create_lorenz_model_observation(
        lorenz, identity_obs_operator, test_consistency=False
    )
    num_model.background_error_sqrt = B_half

    training_data = generate_dataset(
        lorenz,
        x0=lorenz.truth.state_vector[:, -1],
        assimilation_window=window,
        nsamples=nsamples,
    )
    save_data(training_data, target)
