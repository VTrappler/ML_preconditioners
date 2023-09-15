from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
import lightning.pytorch as pl
import torch
import pickle
import sys
import os
from prec_data.background_error import BackgroundError, BackgroundErrorArray


class GaussNewtonDataset(Dataset):
    def __init__(
        self,
        data,
        inverse_background_error_covariance=None,
        inverse_observation_error_covariance=None,
        bck_preconditioned=None,
    ):
        self.data = data
        self.Bmatrix_inv = inverse_background_error_covariance
        self.Bhalf = bck_preconditioned

        if inverse_observation_error_covariance is None:
            x0, f0, t0 = self.data[0]
            m = t0.shape[0]
            self.Rmatrix_inv = np.eye(m)
        else:
            self.Rmatrix_inv = inverse_observation_error_covariance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, forward, tlm = self.data[idx]
        gauss_newton = tlm.T @ self.Rmatrix_inv @ tlm + self.Bmatrix_inv
        if self.Bhalf is not None:
            gauss_newton = self.Bhalf.T @ gauss_newton @ self.Bhalf  # Precondition
        return torch.Tensor(x), torch.Tensor(forward), torch.Tensor(gauss_newton)


class GaussNewtonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dim: int,
        path: str,
        batch_size: int,
        num_workers: int,
        splitting_lengths: list,
        shuffling: bool,
        normalization: bool = False,
        bck_error_covariance_matrix_path: str = None,
        obs_error_covariance_matrix_path: str = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splitting_lengths = splitting_lengths
        self.fractional = isinstance(self.splitting_lengths[0], float)
        self.shuffling = shuffling
        self.path = path
        self.normalization = normalization
        self.bck_covariance_matrix_path = bck_error_covariance_matrix_path
        self.obs_covariance_matrix_path = obs_error_covariance_matrix_path
        self.dim = dim
        super().__init__()

    def setup(self, stage):
        if self.bck_covariance_matrix_path is not None:
            bck_error = BackgroundErrorArray(
                dim=self.dim, path=self.bck_covariance_matrix_path
            )
        else:
            bck_error = BackgroundError(dim=self.dim)

        # if self.obs_covariance_matrix_path is None:
        #     obs_error = ObservationError(dim=self.dim)

        print(bck_error.bck_error_covariance_matrix)
        # print(obs_error.obs_error_covariance_matrix)

        with open(self.path, "rb") as handle:
            gaussnewton_dataset = GaussNewtonDataset(
                pickle.load(handle),
                inverse_background_error_covariance=bck_error.inv_bck_error_covariance_matrix,
                inverse_observation_error_covariance=None,
            )
        total_len = len(gaussnewton_dataset)
        if self.fractional:
            splitting_lengths = [int(total_len * n) for n in self.splitting_lengths]
        else:
            splitting_lengths = self.splitting_lengths

        self.train, self.val, self.test = torch.utils.data.random_split(
            gaussnewton_dataset, splitting_lengths
        )

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
