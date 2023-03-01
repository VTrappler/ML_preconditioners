import os
import torch
from torch import nn
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import torch.nn.functional as F
from torchmetrics.functional import r2_score
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import functools

from typing import List


class EnforcePositivity(nn.Module):
    def __init__(self, mode="huber"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "huber":
            return nn.HuberLoss(reduction="none").forward(x, torch.zeros_like(x))
        elif self.mode == "abs":
            return nn.L1Loss(reduction="none").forward(x, torch.zeros_like(x))
        elif self.mode == "sqr":
            return nn.MSEloss(reduction="none").forward(x, torch.zeros_like(x))


def _eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def eye_like(tensor):
    b = tensor.shape[0]
    eye = torch.empty_like(tensor)
    for i in range(b):
        eye[i, :, :] = torch.eye(tensor.shape[-1])
    return eye

def batch_identity_matrix(batch_size, dimension):
    In = torch.eye(dimension).reshape(-1, dimension, dimension).repeat(batch_size, 1, 1)
    return In

def compose_triangular_matrix(arr: torch.Tensor) -> torch.Tensor:
    arr = torch.atleast_2d(arr)
    k = arr.shape[1]
    n = int((np.sqrt(8 * k + 1) - 1) / 2)
    tri = torch.zeros(len(arr), n, n)
    for j, ar in enumerate(arr):
        l = 0
        for diag, i in enumerate(range(n, -1, -1)):
            tri[j, ...] += torch.diagflat(ar[l : i + l], -diag)
            l += i
    return tri


def construct_model_class(cl, **kwargs):
    class Model(cl):
        __init__ = functools.partialmethod(cl.__init__, **kwargs)

    return Model


def gen_plot(*data):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    n = len(data)
    m = int(np.round(np.sqrt(n)))
    for i, (dat, lab) in enumerate(data):
        ax = plt.subplot(m, m, i + 1)
        im = plt.imshow(dat)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax.set_title(lab)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def bgramschmidt(bV):
    """Batch orthonormalisation of rank vectors, each of dimension ndim
    bV.shape = [batch, ndim, rank]
    """
    def bnormalize(V):
        return V / V.norm(dim=1).view(V.shape[0], 1)
        
    def bproj(bU, bV):
        return (torch.einsum("bn,bn->b", bU, bV) / bU.norm(dim=1)**2).view(bU.shape[0], 1) * bU

    bU = torch.zeros_like(bV)
    bU[:, :, 0] = bnormalize(bV[:, :, 0])
    for i in range(1, bU.shape[-1]):
        bU[:, :, i] = bV[:, :, i].clone()
        for j in range(i):
            bU[:, :, i] -= bproj(bU[:, :, j], bV[:, :, i])
        bU[:, :, i] = bnormalize(bU[:, :, i])
    return bU




class BaseModel(pl.LightningModule):
    def __init__(self, state_dimension: int, config: dict):
        pl.LightningModule.__init__(self)
        # super().__init__()
        self.state_dimension = state_dimension
        if "lr" in config:
            self.lr = config["lr"]
        else:
            self.lr = 1e-4
        self.n_layers = config["n_layers"]
        self.neurons_per_layer = config["neurons_per_layer"]
        self.mse = F.mse_loss
        self.batch_size = config["batch_size"]
        self.identity = batch_identity_matrix(self.batch_size, self.state_dimension)

    def forward(self, x):
        layers = self.layers(x)
        diag = self.to_diag(layers)
        off_diag = self.to_offdiag(layers)
        flat = torch.cat((diag, off_diag), 1)
        return compose_triangular_matrix(flat)

    def construct_full_matrix(self, L_lower):
        y_hat = torch.bmm(
            L_lower, L_lower.mT
        )  # Construct the full matrix from the Cholesky decomposition
        return y_hat

    def weight_histograms_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def gradient_histograms_adder(self):
        global_step = self.global_step
        if global_step % 50 == 0:  # do not make the tb file huge
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.logger.experiment.add_histogram(
                        f"{name}_grad", param.grad, global_step
                    )

    #     def target_imshow(self, outputs, batch_size):
    #         tb = self.logger.experiment
    #         GTG_yhat = torch.zeros((self.batch_size, self.state_dimension, self.state_dimension))
    #         counter = 0
    #         acc_gtg = torch.zeros((self.state_dimension, self.state_dimension))
    #         acc_y_hat = torch.zeros((self.state_dimension, self.state_dimension))
    #         acc_prod = torch.zeros((self.state_dimension, self.state_dimension))
    #         out = outputs[0]
    #         gtg_inv = torch.linalg.inv(out['gtg'][3])
    #         buf = gen_plot((out['gtg'][3] @ out['y_hat'][3], r'$(G^TG) H^{-1}$'),
    #                        # (out['gtg'][3] @ torch.linalg.inv(out['y_hat'][3]), r'$(G^TG) H^{-1}$'),
    #                        (out['gtg'][3], r'$(G^TG)$'),
    #                        (torch.linalg.inv(out['y_hat'][3]), r'$H^{-1}$'),
    #                        (torch.abs(out['gtg'][3] - torch.linalg.inv(out['y_hat'][3])), r'|Difference|')
    #                       )

    #         im = Image.open(buf)
    #         im = transforms.ToTensor()(im)
    #         tb.add_image("identity_matrix", im, global_step=self.current_epoch)

    def _common_step(self, batch, batch_idx, stage):
        x, forw, tlm = batch
        GTG = torch.bmm(
            tlm.mT, tlm
        )  # Get the GN approximation of the Hessian matrix

        x = x.view(x.size(0), -1)
        L_lower = self.forward(x)

        y_hat = self.construct_full_matrix(L_lower)

        identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(len(x), 1, 1)
        )

        product = torch.bmm(y_hat, GTG)  # LL^T @ G^T G

        loss = self.loss(y_hat, product, identity)

        self.log(f"Loss/{stage}_loss", loss)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hat.detach()}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def on_after_backward(self):
        # self.gradient_histograms_adder()
        pass

    def training_epoch_end(self, outputs):
        # self.weight_histograms_adder()
        pass

    def validation_epoch_end(self, outputs):
        # self.weight_histograms_adder()
        batch_size = len(outputs[0]["gtg"])
        # self.target_imshow(outputs, batch_size=batch_size)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # self.log("ptl/val_loss", avg_loss)

    def configure_optimizers(self):
        #     def configure_optimizers(self):
        # optimizer = Adam(self.parameters(), lr=1e-3)
        # scheduler = PlOnPlateau(optimizer, ...)
        # return [optimizer], [scheduler]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]


def construct_MLP(n_in: int, n_hidden: int, n_neurons_per_lay: int, n_out: int) -> nn.Module:
    """Construct a Fully connected MultiLayerPerceptron (LeakyReLU activation)

    :param n_in: dimension of input
    :type n_in: int
    :param n_hidden: number of hidden layers
    :type n_hidden: int
    :param n_neurons_per_lay: neurons per layers
    :type n_neurons_per_lay: int
    :param n_out: dimension of output
    :type n_out: int
    :return: nn.Module
    :rtype: int
    """

    lays = [
        nn.Linear(n_in, n_neurons_per_lay),
        nn.LeakyReLU(),
    ]
    for i in range(n_hidden):
        lays.append(nn.Linear(n_neurons_per_lay, n_neurons_per_lay))
        lays.append(nn.LeakyReLU())
    lays.append(nn.Linear(n_neurons_per_lay, n_out))
    return nn.Sequential(*lays)


def Conv1D_periodic(kernel_size):
    return torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular')

def construct_MLP(n_in: int, n_hidden: int, n_neurons_per_lay: int, n_out: int) -> nn.Module:
    """Construct a Fully connected MultiLayerPerceptron (LeakyReLU activation)

    :param n_in: dimension of input
    :type n_in: int
    :param n_hidden: number of hidden layers
    :type n_hidden: int
    :param n_neurons_per_lay: neurons per layers
    :type n_neurons_per_lay: int
    :param n_out: dimension of output
    :type n_out: int
    :return: nn.Module
    :rtype: int
    """

    lays = [
        nn.Linear(n_in, n_neurons_per_lay),
        nn.LeakyReLU(),
    ]
    for i in range(n_hidden):
        lays.append(nn.Linear(n_neurons_per_lay, n_neurons_per_lay))
        lays.append(nn.LeakyReLU())
    lays.append(nn.Linear(n_neurons_per_lay, n_out))
    return nn.Sequential(*lays)