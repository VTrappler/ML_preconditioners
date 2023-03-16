from typing import Optional
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
import pytorch_lightning as pl
import torch
import pickle
import sys

sys.path.extend(["/", ".."])
import warnings

# from dynamical_system.lorenz_wrapper import LorenzWrapper
from DA_PoC.common.numerical_model import NumericalModel
from DA_PoC.dynamical_systems.lorenz_numerical_model import LorenzWrapper


class TangentLinearDataset(Dataset):
    def __init__(self, data, normalization=True):
        self.data = data
        n_cst = -np.inf

        for _, _, pa in self.data:
            if n_cst <= np.max(pa):
                n_cst = np.max(pa)

        # self.norm_cst = np.max(list(zip(*self.data))[2])
        if normalization:
            self.norm_cst = n_cst
        else:
            self.norm_cst = 1.0
        # self.norm_cst = 1.0
        # print(f"{self.data=}")
        print(f"{self.norm_cst=}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, forward, tlm = self.data[idx]
        # U, S, VT = np.linalg.svd(M)
        # Srm1 = np.zeros_like(S)
        # Srm1[S >= 1e-9] = (S[S >= 1e-9]**-1)
        # SV = Srm1 * VT.T
        return torch.Tensor(x), torch.Tensor(forward), torch.Tensor(tlm / self.norm_cst)


class TangentLinearVectorDataset(Dataset):
    def __init__(self, data):
        self.pairs = data
        # x, dx, G*^T @ G @ dx
        sum = 0
        for i, (_, _, GtGdx) in enumerate(self.pairs):
            sum = sum + np.sum((GtGdx) ** 2)
        self.norm_cst = sum / (i + 1)

        # self.norm_cst = np.max(list(zip(*self.pairs))[2])
        print(f"{self.norm_cst=}")
        # self.norm_cst = 1.0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, dx, GtGdx = self.pairs[idx]
        # U, S, VT = np.linalg.svd(M)
        # Srm1 = np.zeros_like(S)
        # Srm1[S >= 1e-9] = (S[S >= 1e-9]**-1)
        # SV = Srm1 * VT.T
        return torch.Tensor(x), torch.Tensor(dx), torch.Tensor(GtGdx)


class TangentLinearDatasetDummy(Dataset):
    def __init__(self, data):
        self.pairs = data

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        forward, tlm = self.pairs[idx]
        # U, S, VT = np.linalg.svd(M)
        # Srm1 = np.zeros_like(S)
        # Srm1[S >= 1e-9] = (S[S >= 1e-9]**-1)
        # SV = Srm1 * VT.T
        return torch.Tensor(forward), torch.Tensor(tlm)  # , torch.Tensor(SV)


class TangentLinearDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int,
        num_workers: int,
        splitting_lengths: list,
        shuffling: bool,
        normalization: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths
        self.fractional = isinstance(self.splitting_lengths[0], float)
        self.shuffling = shuffling
        self.path = path
        self.normalization = normalization
        super().__init__()

    def setup(self, stage):
        with open(self.path, "rb") as handle:
            tangentlinear_dataset = TangentLinearDataset(
                pickle.load(handle), normalization=self.normalization
            )
        total_len = len(tangentlinear_dataset)
        if self.fractional:
            splitting_lengths = [int(total_len * n) for n in self.splitting_lengths]
        else:
            splitting_lengths = self.splitting_lengths

        self.train, self.val, self.test = torch.utils.data.random_split(
            tangentlinear_dataset, splitting_lengths
        )
        self.norm_cst = tangentlinear_dataset.norm_cst

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


class TangentLinearVectorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int,
        num_workers: int,
        splitting_lengths: list,
        shuffling: bool,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths
        self.fractional = isinstance(self.splitting_lengths[0], float)
        self.shuffling = shuffling
        self.path = path
        super().__init__()

    def setup(self, stage):
        with open(self.path, "rb") as handle:
            tangentlinear_vector_dataset = TangentLinearVectorDataset(
                pickle.load(handle)
            )
        total_len = len(tangentlinear_vector_dataset)
        if self.fractional:
            splitting_lengths = [int(total_len * n) for n in self.splitting_lengths]
        else:
            splitting_lengths = self.splitting_lengths

        self.train, self.val, self.test = torch.utils.data.random_split(
            tangentlinear_vector_dataset, splitting_lengths
        )
        self.norm_cst = tangentlinear_vector_dataset.norm_cst

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


class TLIterableDataset(IterableDataset):
    def __init__(
        self,
        length: int,
        numerical_model: NumericalModel,
        nvectors: int,
        burn: int,
        x0: np.ndarray,
        nobs: int,
        print_init: Optional[str] = None,
    ):
        if print_init is not None:
            print(f"{print_init}")
        super(TLIterableDataset).__init__()
        self.state_dimension = state_dimension
        self.numerical_model = numerical_model
        self.current_x = x0
        self.nvectors = nvectors
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        self.generated = 1
        return self

    def generate_new_state(self, forw):
        self.current_x = forw.reshape(self.state_dimension, -1)[
            :, -1
        ] + 0.1 * np.random.normal(size=self.state_dimension)
        rand_steps = np.random.randint(1, 100)
        # print(f"{rand_steps=}")
        go_ahead = self.numerical_model.forward(self.current_x, rand_steps)
        return go_ahead.reshape(self.state_dimension, -1)[:, -1]

    def __next__(self):
        if self.generated > self.length:
            raise StopIteration
        if self.nvectors == 0:
            dx = np.eye(self.state_dimension)
        else:
            dx = np.random.normal(size=(self.state_dimension, self.nvectors))
        x = self.current_x
        forw, tlm = self.numerical_model.forward_TLM(self.current_x)
        # print(f"forward call")

        tlm = tlm.reshape(
            self.state_dimension * self.lorenz.n_total_obs, self.state_dimension
        )
        self.generated += 1

        self.current_x = self.generate_new_state(forw)

        # print(f"{x=}")
        # print(f"{self.current_x=}")
        return torch.Tensor(x), torch.Tensor(dx), torch.Tensor(tlm.T @ tlm @ dx)


class LorenzTLIterableDataset(IterableDataset):
    def __init__(
        self,
        length: int,
        state_dimension: int,
        nvectors: int,
        burn: int,
        x0: np.ndarray,
        nobs: int,
        print_init: Optional[str] = None,
    ):
        if print_init is not None:
            print(f"{print_init}")
        super(LorenzTLIterableDataset).__init__()
        self.state_dimension = state_dimension
        self.lorenz = LorenzWrapper(self.state_dimension)
        self.lorenz.H = lambda x: x
        if burn > 0:
            x0 = np.random.normal(size=(self.state_dimension,))
            self.lorenz.create_and_burn_truth(burn, x0)
            self.current_x = self.lorenz.truth.state_vector[:, -1]
        else:
            self.current_x = x0
        self.nvectors = nvectors
        self.lorenz.n_total_obs = nobs
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        self.generated = 1
        return self

    def generate_new_state(self, forw):
        self.current_x = forw.reshape(self.state_dimension, -1)[
            :, -1
        ] + 0.1 * np.random.normal(size=self.state_dimension)
        rand_steps = np.random.randint(1, 100)
        # print(f"{rand_steps=}")
        go_ahead = self.lorenz.forward_steps(self.current_x, rand_steps)
        return go_ahead.reshape(self.state_dimension, -1)[:, -1]

    def __next__(self):
        if self.generated > self.length:
            raise StopIteration
        if self.nvectors == 0:
            dx = np.eye(self.state_dimension)
        else:
            dx = np.random.normal(size=(self.state_dimension, self.nvectors))
        x = self.current_x
        forw, tlm = self.lorenz.forward_TLM(self.current_x, return_base=True)
        # print(f"forward call")

        tlm = tlm.reshape(
            self.state_dimension * self.lorenz.n_total_obs, self.state_dimension
        )
        self.generated += 1

        self.current_x = self.generate_new_state(forw)

        # print(f"{x=}")
        # print(f"{self.current_x=}")
        return torch.Tensor(x), torch.Tensor(dx), torch.Tensor(tlm.T @ tlm @ dx)


class LorenzTLVectorIterableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        state_dimension: int,
        nobs: int,
        len_dataset: int,
        nvectors: int,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        # splitting_lengths: list,
        shuffling: bool,
        full_jacobian=False,
    ):
        super(LorenzTLVectorIterableDataModule).__init__()
        self.len_dataset = len_dataset
        self.state_dimension = state_dimension
        self.nobs = nobs
        self.batch_size = batch_size
        self.nvectors = nvectors
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        # self.splitting_lengths = splitting_lengths
        # self.fractional = isinstance(self.splitting_lengths[0], float)
        self.fractional = False
        self.shuffling = shuffling
        self.full_jacobian = full_jacobian
        if self.full_jacobian is True:
            warnings.warn("nvectors is set to 0 to compute the full jacobian matrix")
            self.nvectors = 0
        self.splitting_lengths = [50, 50, 50]
        self.prepare_data_per_node = False
        self._log_hyperparams = False

    def setup(self, stage):
        if self.fractional:
            splitting_lengths = [
                int(self.len_dataset * n) for n in self.splitting_lengths
            ]
        else:
            splitting_lengths = self.splitting_lengths

        self.train_TL_iterable_dataset = LorenzTLIterableDataset(
            length=splitting_lengths[0],
            state_dimension=self.state_dimension,
            nvectors=self.nvectors,
            burn=20_000,
            x0=None,
            nobs=self.nobs,
            print_init="Train dataset initialized",
        )

        self.val_TL_iterable_dataset = LorenzTLIterableDataset(
            length=splitting_lengths[1],
            state_dimension=self.state_dimension,
            nvectors=self.nvectors,
            burn=0,
            x0=self.train_TL_iterable_dataset.current_x,
            nobs=self.nobs,
            print_init="Validation dataset initialized",
        )
        self.test_TL_iterable_dataset = LorenzTLIterableDataset(
            length=splitting_lengths[2],
            state_dimension=self.state_dimension,
            nvectors=self.nvectors,
            burn=0,
            x0=self.train_TL_iterable_dataset.current_x,
            nobs=self.nobs,
            print_init="Test dataset initialized",
        )
        # self.train, self.val, self.test = torch.utils.data.random_split(
        #     TL_iterable_dataset, splitting_lengths
        # )
        # self.norm_cst = TL_iterable_dataset.norm_cst

    def train_dataloader(self):
        return DataLoader(
            self.train_TL_iterable_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_TL_iterable_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_TL_iterable_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
