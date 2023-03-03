from lorenz_wrapper import LorenzWrapper

# from optimization import low_rank_approx
import tqdm
import argparse
import pickle
import numpy as np
from omegaconf import OmegaConf

# def generate_training_pair(x):
#     forw, G = forward_TLM(x, return_base=True)
#     # G, grtlm = grad_TLM(x)
#     Ur, Sr, VrT = low_rank_approx(G, state_dimension)
#     return (forw, Sr), (Ur, Sr, VrT)


def generate_new_state(lorenz, x):
    return lorenz.lorenz_model.integrate(0, x, lorenz.period_assim)[1][
        :, -1
    ] + 0.5 * np.random.normal(size=len(x))


# def generate_training(x0=initial_state, Nobs=100):
#     obs = []
#     x = x0
#     for i in tqdm.trange(Nobs):
#         obs.append(generate_training_pair(x)[0])
#         x = generate_new_state(x)
#     return obs


# def generate_training_pair_VS(x):
#     Geval, G = forward_TLM(x, return_base=True)
#     # G, grtlm = grad_TLM(x)
#     Ur, Sr, VrT = low_rank_approx(G, state_dimension)
#     Srm1 = np.zeros_like(Sr)
#     Srm1[Sr >= 1e-9] = (Sr[Sr >= 1e-9]**-1)
#     return (Geval, Srm1 * VrT.T)

# def generate_training_pair_VS_G(x):
#     forw, G = forward_TLM(x, return_base=True)
#     # G, grtlm = grad_TLM(x)
#     Ur, Sr, VrT = low_rank_approx(G, state_dimension)
#     Srm1 = np.zeros_like(Sr)
#     Srm1[Sr >= 1e-9] = (Sr[Sr >= 1e-9]**-1)
#     return (forw, G, Srm1 * VrT.T, (Ur, Sr, VrT))


def generate_training_pair(lorenz, x):
    return lorenz.forward_TLM(x, return_base=True)


def generate_training_pair_x(lorenz, x):
    forw, tlm = lorenz.forward_TLM(x, return_base=True)
    tlm = tlm.reshape(
        lorenz.state_dimension * lorenz.n_total_obs, lorenz.state_dimension
    )
    return x, forw, tlm


def generate_training(lorenz, x0=None, Nobs=100):
    if x0 is None:
        x0 = lorenz.initial_state
    train = []
    x = x0
    for i in tqdm.trange(Nobs):
        train.append(generate_training_pair(lorenz, x))
        x = generate_new_state(x)
    return train


def generate_training_x(lorenz, x0=None, Nobs=100):
    if x0 is None:
        x0 = lorenz.initial_state
    train = []
    x = x0
    for i in tqdm.trange(Nobs):
        train.append(generate_training_pair_x(lorenz, x))
        x = generate_new_state(lorenz, x)
    return train


def save_data(data, path):
    pickle.dump(data, open(f"{path}", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the training data for the Lorenz model"
    )
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='data_params.yml')
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
        dim = conf.model["dimension"]
        nsamples = conf.model["nsamples"]
        window = conf.model['window']
        target = f"raw_data/{dim}_{window}obs_{nsamples}.pkl"

    else:
        nsamples = args.N
        dim = args.dim
        window = args.n_total_obs
        target = args.target



    print(f"Training tuples to save: {args.N}")
    print(f"into {target}")
    lorenz = LorenzWrapper(dim)
    lorenz.eps = 1e-8
    lorenz.create_and_burn_truth()
    
    lorenz.generate_obs(n_total_obs=window, H=lambda x: x)

    if args.dummy:
        training_data = generate_training_dummy(Nobs=nsamples)
    else:
        training_data = generate_training_x(lorenz, Nobs=nsamples)
    save_data(training_data, args.target)
