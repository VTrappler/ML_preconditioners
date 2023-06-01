import numpy as np
import torch
from torch.utils.data import Dataset
from .data import TangentLinearDataModule
import os


class TangentLinearDatasetMEMMAP(Dataset):
    def __init__(
        self, x_mmap: str, tlm_mmap: str, nsamples: int, dim: int, window: int
    ):
        self.nsamples = nsamples
        self.states = np.memmap(
            x_mmap,
            dtype="float32",
            mode="c",
            shape=(nsamples, dim),
        )
        self.tangent_linear = np.memmap(
            tlm_mmap,
            dtype="float32",
            mode="c",
            shape=(nsamples, dim * window, dim),
        )

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        x = self.states[idx, ...]
        tlm = self.tangent_linear[idx, ...]
        forward = torch.Tensor(np.array([np.nan]))
        return torch.Tensor(x), forward, torch.Tensor(tlm)


class TangentLinearDataModuleMEMMAP(TangentLinearDataModule):
    def __init__(
        self,
        path: str,
        nsamples: int,
        dim: int,
        window: int,
        batch_size: int,
        num_workers: int,
        splitting_lengths: list,
        shuffling: bool,
        normalization: bool = False,
    ):
        super().__init__(
            path, batch_size, num_workers, splitting_lengths, shuffling, normalization
        )
        self.nsamples = nsamples
        self.dim = dim
        self.window = window
        ls = os.listdir(path)
        if ("x.memmap" not in ls) or ("tlm.memmap" not in ls):
            raise RuntimeError(f"No memmap file located in given path: {path}: {ls}")

    def setup(self, stage):
        x_mmap = os.path.join(self.path, "x.memmap")
        tlm_mmap = os.path.join(self.path, "tlm.memmap")
        tangentlinear_dataset = TangentLinearDatasetMEMMAP(
            x_mmap,
            tlm_mmap,
            nsamples=self.nsamples,
            dim=self.dim,
            window=self.window,
        )
        if self.fractional:
            splitting_lengths = [int(self.nsamples * n) for n in self.splitting_lengths]
        else:
            splitting_lengths = self.splitting_lengths

        self.train, self.val, self.test = torch.utils.data.random_split(
            tangentlinear_dataset, splitting_lengths
        )
        self.norm_cst = 1.0

        self.state_test, _, tlm_test = self.test[0]

        self.GN_test = tlm_test.T @ tlm_test
        self.U_test, self.S_test, _ = torch.linalg.svd(self.GN_test)
