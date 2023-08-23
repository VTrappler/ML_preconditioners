import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import TangentLinearDataModule
import os


# eta_slice = slice(None, 64**2)
# u_slice = slice(64**2, 64**2 + 63 * 64)
# v_slice = slice(64**2 + 64 * 63, None)

# def get_control_2D(state):
#     n_batch = state.shape[0]
#     return (
#         state[:, eta_slice].reshape(n_batch, 64, 64),
#         state[:, u_slice].reshape(n_batch, 63, 64),
#         state[:, v_slice].reshape(n_batch, 64, 63),
#     )


# def pad_and_merge(state):
#     n_batch = state.shape[0]
#     eta, u, v = get_control_2D(state)
#     out_vec = torch.empty((n_batch, 3, 64, 64))  # , names=('b', 'f', 'x', 'y')
#     out_vec[:, 0, :, :] = eta
#     out_vec[:, 1, 1:, :] = u
#     out_vec[:, 1, 0, :] = u[:, 0, :]
#     out_vec[:, 2, :, 1:] = v
#     out_vec[:, 2, :, 0] = v[:, :, 0]
#     return out_vec


class SWDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.n_vector = 100
        self.eta_slice = slice(None, 64**2)
        self.u_slice = slice(64**2, 64**2 + 63 * 64)
        self.v_slice = slice(64**2 + 64 * 63, None)

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        filename = f"gn_data_{idx:04d}.npy"
        data = np.load(os.path.join(self.folder, filename))
        linearization_point = data[:, 0]
        z = data[:, 1 : (self.n_vector + 1)]
        Az = data[:, (self.n_vector + 1) :]
        assert z.shape[1] == Az.shape[1]
        return (
            torch.tensor(linearization_point, dtype=torch.float),
            torch.tensor(z, dtype=torch.float),
            torch.tensor(Az, dtype=torch.float),
        )


class SWDataModule(TangentLinearDataModule):
    def __init__(
        self,
        folder: str,
        batch_size: int = 4,
        num_workers: int = 4,
        splitting_lengths: list = [0.8, 0.1, 0.1],
        shuffling: bool = True,
    ):
        super().__init__(
            folder, batch_size, num_workers, splitting_lengths, shuffling, False
        )
        self.nsamples = 2000
        self.dim = 64**2 + 2 * 63 * 64
        self.folder = folder

    def setup(self, stage):
        tangentlinear_dataset = SWDataset(self.folder)
        if self.fractional:
            splitting_lengths = [int(self.nsamples * n) for n in self.splitting_lengths]
        else:
            splitting_lengths = self.splitting_lengths

        self.train, self.val, self.test = torch.utils.data.random_split(
            tangentlinear_dataset, splitting_lengths
        )
        self.norm_cst = 1.0

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
