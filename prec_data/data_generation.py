from DA_PoC.dynamical_systems.lorenz_numerical_model import LorenzWrapper

# from optimization import low_rank_approx
import argparse
import os
import pickle
import numpy as np
from omegaconf import OmegaConf
from tqdm.rich import tqdm, TqdmExperimentalWarning, trange
import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def generate_new_state(lorenz, x):
    return lorenz.lorenz_model.integrate(0, x, lorenz.period_assim)[1][
        :, -1
    ] + 0.0 * np.random.normal(size=len(x))


def generate_training_pair_x(lorenz, x):
    forw, tlm = lorenz.forward_TLM(x, return_base=True)
    tlm = tlm.reshape(
        lorenz.state_dimension * lorenz.n_total_obs, lorenz.state_dimension
    )
    return x, forw, tlm


def generate_training_x(lorenz, x0=None, Nobs=100):
    if x0 is None:
        x0 = lorenz.initial_state
    train = []
    x = x0
    for i in trange(Nobs):
        x, forw, tlm = generate_training_pair_x(lorenz, x)
        x = generate_new_state(lorenz, x)
    return train


def generate_and_save_training_memmap(
    lorenz,
    x_memmap,
    tlm_memmap,
    x0=None,
    Nobs=100,
):
    if x0 is None:
        x0 = lorenz.initial_state
    train = []
    x = x0
    flush_every = 500
    for i in trange(Nobs):
        x, forw, tlm = generate_training_pair_x(lorenz, x)
        x_memmap[i, ...] = x
        tlm_memmap[i, ...] = tlm
        if i % flush_every == 0:
            x_memmap.flush()
            tlm_memmap.flush()
        x = generate_new_state(lorenz, x)
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
        default="config.yml",
    )
    parser.add_argument("-dim", type=int, help="State dimension of the Lorenz model")
    parser.add_argument("-n_total_obs", type=int, help="Number of observations")
    parser.add_argument("-N", type=int, help="Number of training data to save")
    parser.add_argument("-target", type=str, help="target file")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    # def H_nl(x, gamma=4):
    #     return 2 * x + 1

    if len(args.config) > 0:
        conf = OmegaConf.load(args.config)
        dim = conf["model"]["dimension"]
        window = conf["model"]["window"]
        nsamples = conf["data"]["nsamples"]

        target = conf["data"].get(
            "data_path", f"raw_data/{dim}_{window}obs_{nsamples}.pkl"
        )  # if has key else default
        mmap_folder = conf["data"].get("data_folder", None)  # if has key else default

    else:
        nsamples = args.N
        dim = args.dim
        window = args.n_total_obs
        target = args.target

    print(f"Number of training tuples to save: {nsamples}")
    print(f"into {target}")
    lorenz = LorenzWrapper(dim)
    lorenz.eps = 1e-8
    lorenz.create_and_burn_truth()

    lorenz.generate_obs(n_total_obs=window, H=lambda x: x)

    x_memmap = np.memmap(
        os.path.join(mmap_folder, "x.memmap"),
        dtype="float32",
        mode="w+",
        shape=(nsamples, dim),
    )
    tlm_memmap = np.memmap(
        os.path.join(mmap_folder, "tlm.memmap"),
        dtype="float32",
        mode="w+",
        shape=(nsamples, dim * window, dim),
    )

    training_data = generate_and_save_training_memmap(
        lorenz, x_memmap, tlm_memmap, Nobs=nsamples
    )
    # save_data(training_data, target)
