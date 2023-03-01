from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torchmetrics.functional import r2_score

from .base_models import (
    BaseModel,
    compose_triangular_matrix,
    construct_MLP,
    construct_model_class,
    eye_like,
    bgramschmidt
)


class Triangular(BaseModel):
    def __init__(self, state_dimension: int, config: dict) -> None:
        """
        config: dict with keys n_layers, neurons_per_layer, batch_size,
        """
        super().__init__(self, state_dimension, config)
        # super().__init__()

        self.n_out = int(self.state_dimension * (self.state_dimension + 1) / 2)

        self.layers = construct_MLP(
            self.state_dimension, self.n_layers, self.neurons_per_layer, self.n_out
        )

        self.mse = F.mse_loss
        self.batch_size = config["batch_size"]
        self.identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(self.batch_size, 1, 1)
        )
        # print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.layers(x)
        L_lower = compose_triangular_matrix(flat)
        # return Llower + 1e-4 * identity
        return L_lower

    # def loss(self, y_hat, product, identity):
    #     return self.state_dimension * self.mse(product, identity)# + 1.0 * self.mse(torch.zeros_like(y_hat), y_hat)

    def construct_full_matrix(self, L_lower: torch.Tensor) -> torch.Tensor:
        y_hat = torch.bmm(
            L_lower, L_lower.transpose(1, 2)
        )  # Construct the full matrix from the Cholesky decomposition
        # assert y_hat.shape == self.identity.shape
        return y_hat + torch.eye(self.state_dimension).reshape(
            -1, self.state_dimension, self.state_dimension
        ).repeat(y_hat.shape[0], 1, 1)

    def loss(
        self, y_hat: torch.Tensor, product: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        condi = torch.linalg.cond(product).mean()
        return condi


class LowRank(BaseModel):
    def __init__(self, state_dimension: int, rank: int, config: dict) -> None:
        """
        config: dict with keys n_layers, neurons_per_layer, batch_size,
        """
        print(f"{state_dimension=}")
        print(f"{config=}")
        print(f"{rank=}")

        super().__init__(state_dimension=state_dimension, config=config)
        self.rank = rank
        self.n_out = int(rank * state_dimension)

        ## Construction of the common layers
        self.layers = construct_MLP(
            self.state_dimension, self.n_layers, self.neurons_per_layer, self.n_out
        )

        self.mse = F.mse_loss
        self.batch_size = config["batch_size"]
        # print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.layers(x)
        batch_of_low_ranks = flat.reshape(len(x), self.state_dimension, self.rank)
        # torch.nn.functional.normalize(x)
        return torch.nn.functional.normalize(batch_of_low_ranks)

    def construct_full_matrix(self, L_lower):
        LLT = torch.bmm(
            L_lower, L_lower.transpose(1, 2)
        )  # Construct the full matrix from the Cholesky decomposition
        return LLT + eye_like(LLT)

    def construct_inverse_matrix(self, L_lower):
        LTL = torch.bmm(
            L_lower.transpose(1, 2), L_lower
        )
        Ir = eye_like(LTL)
        right = torch.bmm(L_lower, torch.bmm(torch.linalg.inv(Ir + LTL), L_lower.transpose(1, 2)))
        return eye_like(right) - right

    def inference(self, x: torch.Tensor, data_norm=1.0) -> torch.Tensor:
        if not self.training:
            S = self.forward(torch.atleast_2d(x))
        return self.construct_inverse_matrix(S) / data_norm


    def loss(
        self, y_hat: torch.Tensor, product: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        return self.state_dimension * self.mse(
            product, identity
        )  # + 1.0 * self.mse(torch.zeros_like(y_hat), y_hat)

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = torch.bmm(
            tlm.transpose(1, 2), tlm
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)

        S = self.forward(x)
        y_hatinv = self.construct_full_matrix(S)
        # y_hat = self.construct_LMP(S, AS, 1)

        self.identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(len(x), 1, 1)
        )

        # product = torch.bmm(y_hat, GTG) # LL^T @ G^T G
        gram_matrix = torch.bmm(S.transpose(1, 2), S)
        cross_inner_products = torch.sum(torch.triu(gram_matrix, diagonal=1) ** 2)
        # loss = self.loss(y_hat, product, self.identity)
        mse = self.mse(y_hatinv, GTG)
        reg =  cross_inner_products# + self.mse(AtrueS, AS)
        loss = mse + 0.1*reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse", mse)
        self.log(f"Loss/{stage}_regul", reg)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}


class BandMatrix(BaseModel):
    def __init__(self, state_dimension: int, bw: int, config: dict) -> None:
        """
        config: dict with keys n_layers, neurons_per_layer, batch_size,
        """
        super().__init__(self, state_dimension, config)
        self.bw = bw
        self.n_out = int((bw + 1) * (2 * state_dimension - bw) / 2)

        self.layers = construct_MLP(
            self.state_dimension, self.n_layers, self.neurons_per_layer, self.n_out
        )
        self.mse = F.mse_loss
        self.batch_size = config["batch_size"]
        # print(self)

    def construct_band_matrix(self, forw: torch.Tensor) -> torch.Tensor:
        n = self.state_dimension
        batch_size = len(forw)
        mat = torch.zeros((batch_size, n, n))
        for k in range(self.bw + 1):
            if k == 0:
                st = 0
            else:
                st = k * n - k + 1
            for ba in range(batch_size):
                mat[ba, :, :] += torch.diag(forw[ba, st : ((k + 1) * n - k)], -k)
        return mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.layers(x)
        return self.construct_band_matrix(flat)

    def loss(
        self, y_hat: torch.Tensor, product: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        return self.state_dimension * self.mse(product, identity)


class BandMatrixCond(BandMatrix):
    def loss(
        self, y_hat: torch.Tensor, product: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        condi = torch.linalg.cond(product).mean()
        return condi

