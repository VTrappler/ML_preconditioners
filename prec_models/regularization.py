from typing import List, Optional, Tuple
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torchmetrics.functional import r2_score




class Regularization:
    def __init__(self, model, stage) -> None:
        self.pytorch_model = model
        self.stage = stage

    def skew_symmetric(self, S, Sp):
        "S, Sp -> "
        return S.bmm(Sp.mT) - Sp.bmm(S.mT)

    def loss_skew_symmetric(self, S, Sp, coeff):
        skew = self.skew_symmetric(S, Sp)
        reg = coeff * torch.sum(skew ** 2)
        self.pytorch_model.log(f"Reg/{self.stage}_skew_sym", reg)
        return reg

    def gram_matrix(self, S):
        return torch.bmm(S.mT, S)

    def loss_gram_matrix(self, S, coeff):
        upper_sq_mat = coeff * torch.sum(torch.triu(self.gram_matrix(S), diagonal=1)**2)
        self.pytorch_model.log(f"Reg/{self.stage}_gram", upper_sq_mat)
        return upper_sq_mat

    def conjugacy_matrix(self, S, Sp):
        return (S.mT).bmm(Sp)

    def loss_conjugacy_matrix(self, S, Sp, coeff):
        upper_sq_mat = coeff * torch.sum(torch.triu(self.conjugacy_matrix(S, Sp), diagonal=1)**2)
        self.pytorch_model.log(f"Reg/{self.stage}_conjugacy", upper_sq_mat)
        return upper_sq_mat
