from typing import List, Optional, Tuple
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base_models import (
    BaseModel,
    compose_triangular_matrix,
    construct_MLP,
    construct_model_class,
    eye_like,
    bgramschmidt
)

from .convolutional_nn import ConvLayersSVD

from .regularization import Regularization


def outer_prod_rk(q, rk):
    batch_size, state_dimension = q.shape[0], q.shape[1]
    vv = q[:, :, rk].view(batch_size, state_dimension, 1)
    return (vv.bmm(vv.mT))

def construct_deflation(q, coeff, ide):
    H = ide.clone()
    batch_size = q.shape[0]
    for k in range(q.shape[-1]):
        H = H.bmm(ide - coeff[:, k].view(batch_size, 1, 1) * outer_prod_rk(q, k))
    return H

def construct_deflation_backward(q, coeff, ide):
    H = ide.clone()
    batch_size = q.shape[0]
    for k in range(q.shape[-1]):
        H = H.bmm(ide - coeff[:, -k-1].view(batch_size, 1, 1) * outer_prod_rk(q, k))
    return H

def construct_matrix(orth_S, mu):
    """
    Construct matrix M = I + sum (mui_-1)w_iw_i^T
    """

    batch_size, state_dimension, rk = orth_S.shape
    identity = torch.eye(state_dimension).view(1, state_dimension, state_dimension).expand(batch_size, -1, -1)
    sm = 0
    for i in range(rk):
        wi = orth_S[:, :, i].view(batch_size, state_dimension, 1)
        sm += (mu[:, i].view(batch_size, 1, 1)-1) * (wi.bmm(wi.mT))
    return sm + identity


def construct_matrix_USUt(orth_S, diag):
    batch_size, state_dimension, rk = orth_S.shape
    return orth_S.bmm(diag.view(batch_size, rk, 1) * orth_S.mT)

class SVDPrec(BaseModel):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
        datatype: str = "full",
        AS: bool = True,
        n_layers_AS: Optional[int] = None,
    ) -> None:
        """
        config: dict with keys n_layers, neurons_per_layer, batch_size,
        """
        # super().__init__()
        super().__init__(state_dimension, config)
        self.rank = rank
        self.n_out = int(rank * (state_dimension + 1))
        self.AS = AS
        self.datatype = datatype

        ## Construction of the MLP
        ### Construction of the input layer

        self.layers = construct_MLP(
            self.state_dimension,
            self.n_layers,
            self.neurons_per_layer,
            self.n_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x).reshape(len(x), self.state_dimension + 1, self.rank)
        return S

    def construct_approx(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x).reshape(len(x), self.state_dimension + 1, self.rank) # Batch_dimension x (state dimension + 1) x rank
        vi, li = S[:, :-1, :], S[:, -1, :]
        eli = torch.exp(li)
        qi = torch.linalg.qr(vi)[0]
        H = construct_matrix_USUt(qi, eli)
        return H

    def construct_preconditioner(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x).reshape(len(x), self.state_dimension + 1, self.rank) # Batch_dimension x (state dimension + 1) x rank
        vi, li = S[:, :-1, :], S[:, -1, :]
        # eli = torch.exp(-li)
        eli = torch.exp(-li)
        qi = torch.linalg.qr(vi)[0]
        Hm1 = construct_matrix_USUt(qi, eli)
        return Hm1

    def _common_step_full_norm(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = torch.bmm(
            tlm.mT, tlm
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        y_hatinv = self.construct_preconditioner(x)
        GtG_approx = self.construct_approx(x)
        # y_hat = self.construct_LMP(S, AS, 1)

        regul = Regularization(self, stage)
        # loss = self.mse(product, eye_like(product))
        # gram_matrix = torch.bmm(S.mT, S)
        # # skew_sym = y_hatinv.mT - y_hatinv
        # skew_sym = S.bmm(AS.mT) - AS.bmm(S.mT)
        # loss = self.loss(y_hat, product, self.identity)
        mse_approx = self.mse(GtG_approx, GTG)
        # reg =  torch.sum(skew_sym ** 2)# + self.mse(AtrueS, AS)
        loss = mse_approx #+ 0.5 * reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse_approx", mse_approx)

        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}

    def _common_step_iterable(self, batch: Tuple, batch_idx: int, stage: str):
        x, dx, GtGdx = batch
        x = x.view(x.size(0), -1)
        GtG_approx = self.construct_approx(x)
        GtGdx_hat = torch.bmm(GtG_approx, dx)
        mse_approx = self.mse(GtGdx_hat, GtGdx)

        loss = mse_approx #+ 0.5 * reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse_approx", mse_approx)
        return {"loss": loss, "gtg": GtGdx.detach(), "y_hat": GtGdx_hat.detach()}


    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        if self.datatype == "full":
            return self._common_step_full_norm(batch, batch_idx, stage)
        elif self.datatype == 'iterable':
            return self._common_step_iterable(batch, batch_idx, stage)

class SVDConvolutional(SVDPrec):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
    ) -> None:
        """
        config: dict with keys n_layers, neurons_per_layer, batch_size,
        """
        # super().__init__()
        super().__init__(state_dimension, rank, config)
        self.rank = rank
        self.n_out = int(rank * (state_dimension + 1))


        ## Construction of the MLP
        ### Construction of the input layer

        self.layers = ConvLayersSVD(state_dimension, n_latent=rank, kernel_size=3)



class DeflationPrec(BaseModel):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
        datatype: str = "full",
        AS: bool = True,
        n_layers_AS: Optional[int] = None,
    ) -> None:
        """
        config: dict with keys n_layers, neurons_per_layer, batch_size,
        """
        # super().__init__()
        super().__init__(state_dimension, config)
        self.rank = rank
        self.n_out = int(rank * (state_dimension + 1))
        self.AS = AS
        self.datatype = datatype

        ## Construction of the MLP
        ### Construction of the input layer

        self.layers = construct_MLP(
            self.state_dimension,
            self.n_layers,
            self.neurons_per_layer,
            self.n_out,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x).reshape(len(x), self.state_dimension + 1, self.rank)
        return S




    # def construct_preconditioner(self, x: torch.Tensor) -> torch.Tensor:
    #     identity = (
    #         torch.eye(self.state_dimension)
    #         .reshape(-1, self.state_dimension, self.state_dimension)
    #         .repeat(len(x), 1, 1)
    #     )
    #     S = self.forward(x) # Batch_dimension x (state dimension + 1) x rank
    #     vi, li = S[:, :-1, :], S[:, -1, :]
    #     eli = torch.exp(li)
    #     coeffs_H = 1 - (1 / eli)
    #     qi = torch.linalg.qr(vi)[0]
    #     H = construct_deflation(qi, coeffs_H, identity)
    #     return H

    # def construct_inv_preconditioner(self, x: torch.Tensor) -> torch.Tensor:
    #     identity = (
    #         torch.eye(self.state_dimension)
    #         .reshape(-1, self.state_dimension, self.state_dimension)
    #         .repeat(len(x), 1, 1)
    #     )
    #     S = self.forward(x) # Batch_dimension x (state dimension + 1) x rank
    #     vi, li = S[:, :-1, :], S[:, -1, :]
    #     eli = torch.exp(li)
    #     coeffs_B = 1 - eli
    #     qi = torch.linalg.qr(vi)[0]
    #     B = construct_deflation(qi, coeffs_B, identity)
    #     return B

    def construct_preconditioner(self, x: torch.Tensor) -> torch.Tensor:
        S = self.forward(x) # Batch_dimension x (state dimension + 1) x rank
        vi, li = S[:, :-1, :], S[:, -1, :]
        eli = torch.exp(li)
        qi = torch.linalg.qr(vi)[0]
        H = construct_matrix(qi, eli)
        return H

    def construct_inv_preconditioner(self, x: torch.Tensor) -> torch.Tensor:
        S = self.forward(x) # Batch_dimension x (state dimension + 1) x rank
        vi, li = S[:, :-1, :], S[:, -1, :]
        # eli = torch.exp(-li)
        eli = 1/(torch.exp(li))
        qi = torch.linalg.qr(vi)[0]
        Hm1 = construct_matrix(qi, eli)
        return Hm1


    def inference(self, x: torch.Tensor, data_norm=1.0) -> torch.Tensor:
        if not self.training:
            return self.construct_preconditioner(x.view(-1)) / data_norm
        else:
            return None

    def inverse_inference(self, x: torch.Tensor, shift=1.0, data_norm=1.0) -> torch.Tensor:
        if not self.training:
            return self.construct_inv_preconditioner(x.view(-1)) * data_norm
        else:
            return None

    def _common_step_full_norm(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = torch.bmm(
            tlm.mT, tlm
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        y_hatinv = self.construct_inv_preconditioner(x)
        y_hat = self.construct_preconditioner(x)
        # y_hat = self.construct_LMP(S, AS, 1)

        identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(len(x), 1, 1)
        )

        regul = Regularization(self, stage)
        # loss = self.mse(product, eye_like(product))
        # gram_matrix = torch.bmm(S.mT, S)
        # # skew_sym = y_hatinv.mT - y_hatinv
        # skew_sym = S.bmm(AS.mT) - AS.bmm(S.mT)
        # loss = self.loss(y_hat, product, self.identity)
        product = torch.bmm(y_hat, GTG) # LL^T @ G^T G
        product_AHA = torch.bmm(GTG, product)
        mse_inv = self.mse(y_hatinv, GTG)
        mse_prod = self.mse(product, identity)

        mse_AHA = self.mse(product_AHA, GTG)
        # reg =  torch.sum(skew_sym ** 2)# + self.mse(AtrueS, AS)
        loss = mse_AHA #+ 0.5 * reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse_inv", mse_inv)
        self.log(f"Loss/{stage}_mse_prod", mse_prod)
        self.log(f"Loss/{stage}_mse_AHA", mse_AHA)

        # self.log(f"Loss/{stage}_regul", reg)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}

    def _common_step_iterable(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        return LimitedMemoryPrecVectorNorm._common_step(self, batch, batch_idx, stage)

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        if self.datatype == "full":
            return self._common_step_full_norm(batch, batch_idx, stage)
        elif self.datatype == 'iterable':
            return self._common_step_iterable(batch, batch_idx, stage)

