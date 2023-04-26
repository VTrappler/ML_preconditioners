from typing import List, Optional, Tuple
import lightning.pytorch as pl
import torch
from torch import nn

import torch.nn.functional as F

from .base_models import (
    BaseModel,
    construct_MLP,
    construct_model_class,
    eye_like,
    bgramschmidt,
    batch_identity_matrix,
)

from .regularization import Regularization


class LimitedMemoryPrec(BaseModel):
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
        self.n_out = int(rank * state_dimension)
        self.AS = AS
        self.datatype = datatype
        if n_layers_AS is None:
            self.n_layers_AS = self.n_layers
        else:
            self.n_layers_AS = n_layers_AS

        ## Construction of the MLP
        ### Construction of the input layer

        self.layers = construct_MLP(
            self.state_dimension,
            self.n_layers,
            self.neurons_per_layer,
            self.n_out,
        )

        if self.AS:
            self.layersAS = construct_MLP(
                n_in=self.n_out,
                n_hidden=self.n_layers_AS,
                n_neurons_per_lay=self.neurons_per_layer,
                n_out=self.n_out,
            )
        #
        #
        ## Construction of AS LMP
        #
        self.mse = F.mse_loss
        self.batch_size = config["batch_size"]
        self.identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(self.batch_size, 1, 1)
        )
        # print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x)
        return S

    def forward_AS(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x)
        AS = self.layersAS(S)
        return S.reshape(len(x), self.state_dimension, self.rank), AS.reshape(
            len(x), self.state_dimension, self.rank
        )

    def inference(
        self, x: torch.Tensor, shift: float = 1, data_norm: float = 1.0
    ) -> torch.Tensor:
        if not self.training:
            S, AS = self.forward_AS(torch.atleast_2d(x))
        return self.construct_LMP(S, AS, shift=shift) / data_norm

    def inverse_inference(
        self, x: torch.Tensor, shift: float = 1.0, data_norm: float = 1.0
    ) -> torch.Tensor:
        S, AS = self.forward_AS(torch.atleast_2d(x))
        return self.construct_invLMP(S, AS, shift=shift) * data_norm

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = self._construct_gaussnewtonmatrix(
            batch
        )  # Get the GN approximation of the Hessian matrix

        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        if not self.AS:
            S = self.forward(x)
            AtrueS = torch.bmm(GTG, S)
            y_hatinv = self.construct_invLMP(S, AtrueS, 1)
        if self.AS:
            S, AS = self.forward_AS(x)
            y_hatinv = self.construct_invLMP(S, AS, 1)
        # y_hat = self.construct_LMP(S, AS, 1)

        self.identity = batch_identity_matrix(x.shape[0], self.state_dimension)

        # product = torch.bmm(y_hat, GTG) # LL^T @ G^T G
        regul = Regularization(self, stage)

        # loss = self.loss(y_hat, product, self.identity)
        mse = self.mse(y_hatinv, GTG)  # + self.mse(AtrueS, AS)
        # skew_sym = S.bmm(AS.mT) - AS.bmm(S.mT)
        reg = regul.loss_skew_symmetric(S, AS, 1.0)  # + self.mse(AtrueS, AS)

        loss = mse
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse", mse)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}

    def construct_LMP(
        self, S: torch.Tensor, AS: torch.Tensor, shift: float = 1.0
    ) -> torch.Tensor:
        """Construct a LMP using S, and AS, ie approx of A^-1

        :param S: r Column vectors (batch x n x r)
        :type S: torch.Tensor
        :param AS: A @ column vectors (batch x n x r)
        :type AS: torch.Tensor
        :param shift: shift factor, defaults to 1.0
        :type shift: float
        :return: Preconditioner (~A^-1)
        :rtype: torch.Tensor
        """
        In = batch_identity_matrix(S.shape[0], self.state_dimension)
        StASm1 = torch.linalg.inv(torch.bmm(S.mT, AS))
        left = In - torch.bmm(torch.bmm(S, StASm1), AS.mT)
        mid = In - torch.bmm(torch.bmm(AS, StASm1), S.mT)
        right = torch.bmm(torch.bmm(S, StASm1), S.mT)
        H = torch.bmm(left, mid) + shift * right
        return H

    def construct_invLMP(
        self, S: torch.Tensor, AS: torch.Tensor, shift: float = 1.0
    ) -> torch.Tensor:
        """Construct the inverse of the preconditioner, ie approx of A

        :param S: r Column vectors (batch x n x r)
        :type S: torch.Tensor
        :param AS: A @ column vectors (batch x n x r)
        :type AS: torch.Tensor
        :param shift: shift factor, defaults to 1.0
        :type shift: float, optional
        :return: Inverse of preconditioner
        :rtype: torch.Tensor
        """
        In = batch_identity_matrix(S.shape[0], self.state_dimension)
        StASm1 = torch.linalg.inv(torch.bmm(S.mT, AS))
        B = (
            In
            + (1 / shift) * torch.bmm(torch.bmm(AS, StASm1), AS.mT)
            - torch.bmm(
                torch.bmm(S, torch.linalg.inv(torch.bmm(S.mT, S))),
                S.mT,
            )
        )
        return B


class LMP(LimitedMemoryPrec):
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
        super().__init__(state_dimension, rank, config, "full", True, None)
        self.rank = rank
        self.n_out = int(rank * state_dimension)
        self.n_layers_AS = self.n_layers

        ## Construction of the MLP
        ### Construction of the input layer

        self.layers = construct_MLP(
            self.state_dimension,
            self.n_layers,
            self.neurons_per_layer,
            self.n_out,
        )
        self.layersAS = construct_MLP(
            n_in=self.n_out,
            n_hidden=self.n_layers_AS,
            n_neurons_per_lay=self.neurons_per_layer,
            n_out=self.n_out,
        )
        #
        #
        ## Construction of AS LMP
        #
        self.mse = F.mse_loss
        self.batch_size = config["batch_size"]
        self.identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(self.batch_size, 1, 1)
        )
        # print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x)
        AS = self.layersAS(S).reshape(len(x), self.state_dimension, self.rank)
        return torch.stack(
            [S.reshape(len(x), self.state_dimension, self.rank), AS], axis=-1
        )

    def inference(
        self, x: torch.Tensor, shift: float = 1, data_norm: float = 1.0
    ) -> torch.Tensor:
        if not self.training:
            output = self.forward(torch.atleast_2d(x))
            S, AS = output[..., 0], output[..., 1]
        return self.construct_LMP(S, AS, shift=shift) / data_norm

    def inverse_inference(
        self, x: torch.Tensor, shift: float = 1.0, data_norm: float = 1.0
    ) -> torch.Tensor:
        output = self.forward(torch.atleast_2d(x))
        S, AS = output[..., 0], output[..., 1]
        return self.construct_invLMP(S, AS, shift=shift) * data_norm

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = self._construct_gaussnewtonmatrix(
            batch
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        output = self.forward(torch.atleast_2d(x))
        S, AS = output[..., 0], output[..., 1]
        y_hatinv = self.construct_invLMP(S, AS, 1)
        # y_hat = self.construct_LMP(S, AS, 1)

        self.identity = batch_identity_matrix(x.shape[0], self.state_dimension)

        # product = torch.bmm(y_hat, GTG) # LL^T @ G^T G
        regul = Regularization(self, stage)

        # loss = self.loss(y_hat, product, self.identity)
        mse = self.mse(y_hatinv, GTG)  # + self.mse(AtrueS, AS)
        reg = regul.loss_skew_symmetric(S, AS, 1.0)  # + self.mse(AtrueS, AS)

        loss = mse + reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse", mse)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}


class LimitedMemoryPrecRegularized(LimitedMemoryPrec):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
        datatype: str = "full",
        AS: bool = True,
        n_layers_AS: Optional[int] = None,
    ) -> None:
        super().__init__(state_dimension, rank, config, datatype, AS, n_layers_AS)

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = self._construct_gaussnewtonmatrix(batch)
        # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        if not self.AS:
            S = self.forward(x)
            AtrueS = torch.bmm(GTG, S)
            y_hatinv = self.construct_invLMP(S, AtrueS, 1)
        if self.AS:
            S, AS = self.forward_AS(x)
            y_hatinv = self.construct_invLMP(S, AS, 1)
        # y_hat = self.construct_LMP(S, AS, 1)
        # product = torch.bmm(y_hat, GTG) # LL^T @ G^T G
        gram_matrix = torch.bmm(S.mT, S)
        cross_inner_products = torch.sum(torch.triu(gram_matrix, diagonal=1) ** 2)
        # loss = self.loss(y_hat, product, self.identity)
        mse = self.mse(y_hatinv, GTG)
        reg = cross_inner_products  # + self.mse(AtrueS, AS)
        loss = mse + 0.1 * reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse", mse)
        self.log(f"Loss/{stage}_regul", reg)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}


class LimitedMemoryPrecSym(LimitedMemoryPrec):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
        datatype: str = "full",
        AS: bool = True,
        n_layers_AS: Optional[int] = None,
    ) -> None:
        super().__init__(state_dimension, rank, config, datatype, AS, n_layers_AS)

    def forward_AS(self, x: torch.Tensor) -> torch.Tensor:
        S = self.layers(x)
        S_ortho = torch.linalg.qr(S.view(len(x), self.state_dimension, self.rank))[0]
        AS = self.layersAS(S)
        AS = AS.reshape(len(x), self.state_dimension, self.rank)
        return S_ortho, AS

    def _common_step_full_norm(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = self._construct_gaussnewtonmatrix(
            batch
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        if not self.AS:
            S = self.forward(x)
            AtrueS = torch.bmm(GTG, S)
            y_hatinv = self.construct_invLMP(S, AtrueS, 1)
        if self.AS:
            S, AS = self.forward_AS(x)
            y_hatinv = self.construct_invLMP(S, AS, 1)
        # y_hat = self.construct_LMP(S, AS, 1)

        regul = Regularization(self, stage)

        # product = torch.bmm(y_hat, GTG) # LL^T @ G^T G
        # gram_matrix = torch.bmm(S.mT, S)
        # skew_sym = y_hatinv.mT - y_hatinv
        # skew_sym = S.bmm(AS.mT) - AS.bmm(S.mT)
        # SSpr = S.bmm(AS.mT) - self.identity
        # off_diag_orth = torch.sum(torch.triu(S.mT.bmm(AS), diagonal=1) ** 2)
        # loss = self.loss(y_hat, product, self.identity)
        mse = self.mse(y_hatinv, GTG)
        # reg =  torch.sum(skew_sym ** 2)# + self.mse(AtrueS, AS)
        reg = regul.loss_conjugacy_matrix(S, AS, coeff=1e-3)
        reg += regul.loss_conjugacy_matrix(AS, S, coeff=1e-3)
        # reg += regul.loss_gram_matrix(S, coeff=1)
        loss = mse + reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse", mse)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}

    def _common_step_iterable(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        return LimitedMemoryPrecVectorNorm._common_step(self, batch, batch_idx, stage)

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        if self.datatype == "full":
            return self._common_step_full_norm(batch, batch_idx, stage)
        elif self.datatype == "iterable":
            return self._common_step_iterable(batch, batch_idx, stage)


class LMPLinOp(LimitedMemoryPrec):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
        datatype: str = "full",
        n_layers_AS: Optional[int] = None,
    ) -> None:
        super().__init__(state_dimension, rank, config, datatype, True, n_layers_AS)

        ## Construction of the MLP
        ### Construction of the input layer

        self.layers = construct_MLP(
            self.state_dimension,
            self.n_layers,
            self.neurons_per_layer,
            self.n_out,
        )

        self.layers_linear_op = construct_MLP(
            self.state_dimension,
            self.n_layers,
            self.neurons_per_layer,
            int(rank * (state_dimension + 0)),  # self.nout
        )

    def forward_S(self, x: torch.Tensor) -> torch.Tensor:
        S_ = self.layers(x).reshape(len(x), self.state_dimension, self.rank)
        S = torch.linalg.qr(S_)[0]
        return S

    # def forward_linop(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.layers_linear_op(x).reshape(len(x), self.state_dimension + 1, self.rank)

    def forward_linop(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_linear_op(x).reshape(
            len(x), self.state_dimension + 0, self.rank
        )

    def lowrank_linear_operator(self, x: torch.Tensor) -> torch.Tensor:
        L = self.forward_linop(x)
        LLT = L.bmm(L.mT)
        return LLT + 1e-3 * eye_like(LLT)

    def lowrank_linear_operator_eigendec(self, x: torch.Tensor) -> torch.Tensor:
        vili = self.forward_linop(x)
        vi, li = vili[:, :-1, :], vili[:, -1, :]
        eli = torch.exp(li) + 1
        qi = torch.linalg.qr(vi)[0]
        Atilde = construct_matrix(qi, eli)
        return Atilde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.forward_S(x)
        Atilde = self.lowrank_linear_operator(x)
        # Atilde = self.lowrank_linear_operator_eigendec(x)
        return torch.stack([S, Atilde.bmm(S)], axis=-1)

    def _common_step_full_norm(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, forw, tlm = batch
        GTG = self._construct_gaussnewtonmatrix(
            batch
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        x = x.view(x.size(0), -1)
        output = self.forward(x)
        S, AS = output[..., 0], output[..., 1]

        y_hatinv = self.construct_invLMP(S, AS, 1)
        y_hat = self.construct_LMP(S, AS, 1)

        identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(len(x), 1, 1)
        )
        regul = Regularization(self, stage)

        product_left = torch.bmm(y_hat, GTG)  # LL^T @ G^T G
        product_right = torch.bmm(GTG, y_hat)  # LL^T @ G^T G
        product_AHA = torch.bmm(product_right, GTG)  #

        # gram_matrix = torch.bmm(S.mT, S)
        # skew_sym = y_hatinv.mT - y_hatinv

        # lr_lin_op = self.lowrank_linear_operator(x)
        # skew_sym = S.bmm(AS.mT) - AS.bmm(S.mT)
        # loss = self.loss(y_hat, product, self.identity)
        # mse = self.mse(y_hatinv, GTG)
        mse_inv = self.mse(y_hatinv, GTG)
        mse_prod = self.mse(product_right, identity)
        mse_AHA = self.mse(product_AHA, GTG)
        # reg =  torch.sum(skew_sym ** 2)# + self.mse(AtrueS, AS)
        # reg_lr_GtG = self.mse(lr_lin_op, GTG)
        reg = regul.loss_conjugacy_matrix(S, AS, 1e-3)
        loss = mse_AHA + reg  # + reg_lr_GtG
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse_prod", mse_prod)
        self.log(f"Loss/{stage}_mse_inv", mse_inv)
        self.log(f"Loss/{stage}_mse_AHA", mse_AHA)

        # self.log(f"Loss/{stage}_reg_lr_gtg", reg_lr_GtG)
        # self.log(f"Loss/{stage}_regul", reg)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": y_hatinv.detach()}

    def _common_step_randomvectors(
        self, batch: Tuple, batch_idx: int, stage: str
    ) -> dict:
        x, forw, tlm = batch
        GTG = self._construct_gaussnewtonmatrix(
            batch
        )  # Get the GN approximation of the Hessian matrix
        # Add here the addition
        ## GTG + B^-1.reshape(-1, self.state_dimension, self.state_dimension)
        z_random = torch.randn(size=(len(x), self.state_dimension, self.n_rnd_vectors))
        x = x.view(x.size(0), -1)
        output = self.forward(x)
        S, AS = output[..., 0], output[..., 1]

        H_inv = self.construct_invLMP(S, AS, 1)
        H = self.construct_LMP(S, AS, 1)
        H_Z = H @ z_random
        H_inv_Z = H_inv @ z_random
        GTG_Z = GTG @ z_random
        identity = (
            torch.eye(self.state_dimension)
            .reshape(-1, self.state_dimension, self.state_dimension)
            .repeat(len(x), 1, 1)
        )
        regul = Regularization(self, stage)

        product_left = torch.bmm(H, GTG)  # LL^T @ G^T G
        product_right = torch.bmm(GTG, H)  # LL^T @ G^T G
        product_AHA = torch.bmm(product_right, GTG)  #

        # gram_matrix = torch.bmm(S.mT, S)
        # skew_sym = Hinv.mT - y_hatinv

        # lr_lin_op = self.lowrank_linear_operator(x)
        # skew_sym = S.bmm(AS.mT) - AS.bmm(S.mT)
        # loss = self.loss(y_hat, product, self.identity)
        # mse = self.mse(y_hatinv, GTG)
        mse_inv = self.mse(H_inv_Z, GTG_Z)
        mse_prod = self.mse(product_right @ z_random, identity @ z_random)
        mse_AHA = self.mse(product_AHA @ z_random, GTG_Z)
        # reg =  torch.sum(skew_sym ** 2)# + self.mse(AtrueS, AS)
        # reg_lr_GtG = self.mse(lr_lin_op, GTG)
        reg = regul.loss_conjugacy_matrix(S, AS, 100)
        loss = mse_prod  # +  mse_inv +reg  # + reg_lr_GtG
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse_prod", mse_prod)
        self.log(f"Loss/{stage}_mse_inv", mse_inv)
        self.log(f"Loss/{stage}_mse_AHA", mse_AHA)

        # self.log(f"Loss/{stage}_reg_lr_gtg", reg_lr_GtG)
        # self.log(f"Loss/{stage}_regul", reg)
        return {"loss": loss, "gtg": GTG.detach(), "y_hat": H_inv.detach()}

    def _common_step_iterable(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        return LimitedMemoryPrecVectorNorm._common_step(self, batch, batch_idx, stage)

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        if (self.n_rnd_vectors is not None) and (self.n_rnd_vectors != 0):
            return self._common_step_randomvectors(batch, batch_idx, stage)
        else:
            return self._common_step_full_norm(batch, batch_idx, stage)


class LimitedMemoryPrecVectorNorm(LimitedMemoryPrec):
    def __init__(
        self,
        state_dimension: int,
        rank: int,
        config: dict,
        AS: bool = True,
        n_layers_AS: Optional[int] = None,
    ) -> None:
        super().__init__(state_dimension, rank, config, AS, n_layers_AS)

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> dict:
        x, dx, GtGdx = batch
        x = x.view(x.size(0), -1)
        S, AS = self.forward_AS(x)
        y_hatinv = torch.bmm(self.construct_invLMP(S, AS, 1), dx)

        gram_matrix = torch.bmm(S.mT, S)
        cross_inner_products = torch.sum(torch.triu(gram_matrix, diagonal=1) ** 2)
        # loss = self.loss(y_hat, product, self.identity)
        reg = cross_inner_products  # + self.mse(AtrueS, AS)

        mse = self.mse(y_hatinv, GtGdx)
        reg = cross_inner_products  # + self.mse(AtrueS, AS)
        loss = mse + 0.1 * reg
        self.log(f"Loss/{stage}_loss", loss)
        self.log(f"Loss/{stage}_mse", mse)
        self.log(f"Loss/{stage}_regul", reg)
        return {"loss": loss, "gtg": GtGdx.detach(), "y_hat": y_hatinv.detach()}

    def inference(self, x: torch.Tensor, shift=1, data_norm=1.0) -> torch.Tensor:
        if not self.training:
            S, AS = self.forward_AS(torch.atleast_2d(x))
            return self.construct_LMP(S, AS, shift=shift) / np.sqrt(data_norm)
        else:
            return None
