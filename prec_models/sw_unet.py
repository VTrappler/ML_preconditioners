from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import lightning.pytorch as pl
import torch
from torch import nn
import pickle

import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base_models import (
    BaseModel,
    construct_MLP,
    construct_model_class,
    eye_like,
    low_rank_construction,
)

from .convolutional_nn import ParamSigmoid

from .regularization import Regularization
from .unet_base import UNet as AttentionUNet

controlsize = 64**2 + 2 * 63 * 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eta_slice = slice(None, 64**2)
u_slice = slice(64**2, 64**2 + 63 * 64)
v_slice = slice(64**2 + 64 * 63, None)

import math


class UNet(pl.LightningModule):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.activation = nn.LeakyReLU()
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.1)
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # (channels x h x w)
        # input: 3x64x64
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 64x64x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 64x64x64
        self.norm1 = nn.LayerNorm([64, 64, 64])
        self.enc_layers1 = nn.Sequential(
            self.e11,
            self.norm1,
            self.activation,
            self.dropout,
            self.e12,
            self.norm1,
            self.activation,
            self.dropout,
        )
        self.pool1 = self.pooling  # output: 64x32x32

        # input: 64x32x32
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 128x32x32
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 128x32x32
        self.norm2 = nn.LayerNorm([128, 32, 32])
        self.enc_layers2 = nn.Sequential(
            self.e21,
            self.norm2,
            self.activation,
            self.dropout,
            self.e22,
            self.norm2,
            self.activation,
            self.dropout,
        )
        self.pool2 = self.pooling  # output: 128x16x16

        # input: 128x16x16
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 256x16x16
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 256x16x16
        self.norm3 = nn.LayerNorm([256, 16, 16])
        self.enc_layers3 = nn.Sequential(
            self.e31,
            self.norm3,
            self.activation,
            self.dropout,
            self.e32,
            self.norm3,
            self.activation,
            self.dropout,
        )
        self.pool3 = self.pooling  # output: 256x8x8

        # input: 256x8x8
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 512x8x8
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 512x8x8
        self.enc_layers4 = nn.Sequential(
            self.e41, self.activation, self.e42, self.activation
        )
        self.pool4 = self.pooling  # output: 512x4x4

        # input: 512x4x4
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output:1024x4x4
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 1024x4x4
        self.enc_layers5 = nn.Sequential(
            self.e51,
            self.activation,
            self.dropout,
            self.e52,
            self.activation,
            self.dropout,
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_layers1 = nn.Sequential(
            self.d11,
            self.activation,
            self.dropout,
            self.d12,
            self.activation,
            self.dropout,
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_layers2 = nn.Sequential(
            self.d21,
            self.activation,
            self.dropout,
            self.d22,
            self.activation,
            self.dropout,
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_layers3 = nn.Sequential(
            self.d31,
            self.activation,
            self.dropout,
            self.d32,
            self.activation,
            self.dropout,
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 3 * rank, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(3 * rank, 3 * rank, kernel_size=3, padding=1)
        self.dec_layers4 = nn.Sequential(
            self.d41,
            self.activation,
            self.dropout,
            self.d42,
            self.activation,
            self.dropout,
        )

        # Output layer
        self.final_dense_layer = nn.Linear(3 * 64**2, controlsize + 1)
        # self.final_dense_layer2 = nn.Linear(3 * 64**2, controlsize + 1)
        # self.final_layers = nn.Sequential(
        #     self.final_dense_layer,
        #     self.activation,
        #     self.final_dense_layer2,
        # )
        self.sigmoid = ParamSigmoid(0, 12)
        print(self)

    def __repr__(self):
        total_size = 0
        for i in self.parameters():
            total_size += i.element_size() * i.nelement()
        return f"{self.__class__.__name__}: {total_size / 1e9} Gb"

    def forward_unet(self, x):
        # Encoder
        xe12 = self.enc_layers1(x)
        xp1 = self.pool1(xe12)
        xe22 = self.enc_layers2(xp1)
        xp2 = self.pool2(xe22)
        xe32 = self.enc_layers3(xp2)
        xp3 = self.pool3(xe32)
        xe42 = self.enc_layers4(xp3)
        xp4 = self.pool4(xe42)
        xe52 = self.enc_layers5(xp4)

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd12 = self.dec_layers1(xu11)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd22 = self.dec_layers2(xu22)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd32 = self.dec_layers3(xu33)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd42 = self.dec_layers4(xu44)
        # Output layer
        return xd42

    def forward(self, x):
        x = self.forward_unet(x)
        x = x.reshape(-1, int(self.rank), 3 * 64**2)
        outputs = self.final_dense_layer(x).transpose(-1, -2)
        vecs, vals = outputs[:, :-1, :], outputs[:, -1, :]
        vals = self.sigmoid(vals)
        return torch.concat([vecs, vals[:, None, :]], axis=1)


class TransUNet(pl.LightningModule):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.transUnet = AttentionUNet(
            in_channel=3,
            inner_channel=30,
            out_channel=3 * self.rank,
            res_blocks=5,
            attn_res=[2, 4, 8],
        )
        self.final_dense_layer = nn.Linear(3 * 64**2, controlsize + 1)
        self.sigmoid = ParamSigmoid(0, 12)
        print(self)

    def __repr__(self):
        total_size = 0
        for i in self.parameters():
            total_size += i.element_size() * i.nelement()
        return f"{self.__class__.__name__}: {total_size / 1e9} Gb"

    def forward(self, x):
        x = self.transUnet(x)
        x = x.reshape(-1, int(self.rank), 3 * 64**2)
        outputs = self.final_dense_layer(x).transpose(-1, -2)
        vecs, vals = outputs[:, :-1, :], outputs[:, -1, :]
        vals = self.sigmoid(vals)
        return torch.concat([vecs, vals[:, None, :]], axis=1)


def get_control_2D(state):
    n_batch = state.shape[0]
    return (
        state[:, eta_slice].reshape(n_batch, 64, 64),
        state[:, u_slice].reshape(n_batch, 63, 64),
        state[:, v_slice].reshape(n_batch, 64, 63),
    )


def pad_and_merge(state):
    """
    from batch of state vectors -> batch x channel x width x height
    """
    n_batch = state.shape[0]
    eta, u, v = get_control_2D(state)
    out_vec = torch.empty(
        (n_batch, 3, 64, 64), device=device
    )  # , names=('b', 'f', 'x', 'y')
    out_vec[:, 0, :, :] = eta
    out_vec[:, 1, 1:, :] = u
    out_vec[:, 1, 0, :] = u[:, 0, :]
    out_vec[:, 2, :, 1:] = v
    out_vec[:, 2, :, 0] = v[:, :, 0]
    return out_vec  # batch x channel x width x height


class SW_UNet(BaseModel):
    def __init__(self, state_dimension, rank, config):
        self.controlsize = 64**2 + 2 * 63 * 64
        super().__init__(controlsize, config)
        self.rank = rank
        self.n_out = int(rank * (self.controlsize + 1))
        self.layers = UNet(rank)
        self.normalizer = torch.tensor([15.0, 0.03, 0.03], device=device).reshape(
            1, 3, 1, 1
        )
        # with open(
        #     f"/home/data/data_data_assimilation/shallow_water/eigenvalues/eigenvalues_mean_state.pkl",
        #     "rb",
        # ) as f:
        #     _mean_state_eigs = pickle.load(f)
        # mean_state_eigs = _mean_state_eigs["mean_state_eigvals"][1][:, ::-1].copy()
        # self.mean_state_eigvectors = torch.tensor(
        #     mean_state_eigs[:, : self.rank], dtype=torch.float
        # ).to("cuda:0")

    def normalize_state(self, x_img):
        return x_img / self.normalizer

    def forward(self, x):
        x = pad_and_merge(x)
        x = self.normalize_state(x)
        return self.layers(x)

    def construct_approx(self, x):
        nn_output = self.forward(x)
        vi, log_li = nn_output[:, :-1, :], nn_output[:, -1, :]
        singular_values = torch.exp(log_li)
        singular_vectors = torch.linalg.qr(vi)[0]
        H = low_rank_construction(singular_vectors, singular_values)
        return H, singular_values, vi, singular_vectors

    def construct_preconditioner(self, x: torch.Tensor) -> torch.Tensor:
        nn_output = self.forward(x)
        vi, log_li = nn_output[:, :-1, :], nn_output[:, -1, :]
        singular_values = torch.exp(-log_li)
        singular_vectors = torch.linalg.qr(vi)[0]
        Hm1 = low_rank_construction(singular_vectors, singular_values)
        return Hm1

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str):
        x, z, Az = batch
        # GTG = self._construct_gaussnewtonmatrix(batch)
        # Get the GN approximation of the Hessian matrix
        x = x.view(x.size(0), -1)
        # y_hatinv = self.construct_preconditioner(x)
        GtG_approx, singular_values, vi, singular_vectors = self.construct_approx(x)
        GtG_approx_Z = GtG_approx @ z
        # y_hat = self.construct_LMP(S, AS, 1)

        regul = Regularization(self, stage)

        mse_approx = self.mse(GtG_approx_Z, Az)
        norm_Az = self.mse(0 * GtG_approx_Z, Az)
        # eigvecs_regul = self.mse(
        #     singular_vectors,
        #     self.mean_state_eigvectors.reshape(-1, self.controlsize, self.rank),
        # )
        # reg =  torch.sum(skew_sym ** 2)# + self.mse(AtrueS, AS)

        # vi = torch.nn.functional.normalize(vi, dim=1)

        ortho_reg = regul.loss_gram_matrix(vi, coeff=1e6)
        relative_loss = mse_approx / norm_Az
        loss = mse_approx + ortho_reg  # + 1e9 * eigvecs_regul
        self.log(f"Loss/{stage}_loss", loss, prog_bar=True)
        self.log(f"Info/{stage}_norm_Az", norm_Az)
        # self.log(f"Loss/{stage}_eigvecs_loss", 1e9 * eigvecs_regul)
        self.log(f"Loss/{stage}_rel_loss", relative_loss)
        self.log(f"Loss/{stage}_mse_approx", mse_approx)
        return {"loss": loss}


class SW_TransUNet(SW_UNet):
    def __init__(self, state_dimension, rank, config):
        self.controlsize = 64**2 + 2 * 63 * 64
        super().__init__(state_dimension, rank, config)
        print(f"{self.normalizer.device=}")
        self.layers = TransUNet(rank)


class ConvSW(nn.Module):
    def __init__(self, rk, hidden_channels, n_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.rk = rk
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 64x64x64
        self.c2 = nn.Conv2d(64, 3 * rk, kernel_size=3, padding=1)  # output: 64x64x64
        self.c3 = nn.Conv3d(3, hidden_channels, kernel_size=3, padding=1)
        self.c4 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.c5 = nn.Conv3d(hidden_channels, 3, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        # self.sigmoid = ParamSigmoid(0, 12)
        self.conv2d = nn.Sequential(
            self.c1,
            self.activation,
            self.c2,
            self.activation,
        )
        self.conv3d = nn.Sequential(
            self.c3,
            *[
                nn.Sequential(
                    *[
                        nn.Conv3d(
                            hidden_channels, hidden_channels, kernel_size=3, padding=1
                        ),
                        self.activation,
                        self.dropout,
                    ]
                )
                for _ in range(self.n_layers)
            ],
        )
        self.linear = nn.Linear(3 * 64**2, controlsize + 1)

    def forward(self, x):
        x = self.conv2d(x).reshape(x.shape[0], 3, self.rk, 64, 64)
        x = self.conv3d(x)
        x = self.c5(x)
        x = x.transpose(1, 2).reshape(x.shape[0], self.rk, -1)
        outputs = self.linear(x).transpose(-1, -2)
        vecs, vals = outputs[:, :-1, :], outputs[:, -1, :]
        vals = vals.exp()
        return torch.concat([vecs, vals[:, None, :]], axis=1)


class SW_Conv(SW_UNet):
    def __init__(self, state_dimension, rank, config):
        self.controlsize = 64**2 + 2 * 63 * 64
        super().__init__(controlsize, rank, config)
        self.rank = self.rank
        self.n_out = int(self.rank * (self.controlsize + 1))
        self.layers = ConvSW(self.rank, 128, 5)


model_list = [SW_UNet, SW_Conv, SW_TransUNet]
model_classes = {cl.__name__: cl for cl in model_list}
