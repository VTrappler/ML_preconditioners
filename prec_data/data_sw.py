import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import TangentLinearDataModule
import os


class SWDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.n_vector = 100
        self.eta_slice = slice(None, 64**2)
        self.u_slice = slice(64**2, 64**2 + 63 * 64)
        self.v_slice = slice(64**2 + 64 * 63, None)
        self.length = len([st for st in os.listdir(self.folder) if st.endswith('.npy')])

    def __len__(self):
        return self.length

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
        self.nsamples = 1000
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
