from prec_data.data import CholeskyDataModule
from prec_models.models import BandMatrix, LimitedMemoryPrec
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import argparse
import functools
from pytorch_lightning.loggers import TensorBoardLogger


def construct_model_class(cl, **kwargs):
    class dummy(cl):
        __init__ = functools.partialmethod(cl.__init__, **kwargs)

    return dummy


torch_model = construct_model_class(LimitedMemoryPrec, rank=4)


def main(state_dimension: int, data_path: str):
    n_input = state_dimension
    print(f"{n_input=}")
    bw = 2
    config = {"n_layers": 2, "neurons_per_layer": 64, "batch_size": 5}

    model = torch_model(state_dimension=state_dimension, config=config)
    datamodule = CholeskyDataModule(
        path=data_path,
        batch_size=config["batch_size"],
        num_workers=4,
        splitting_lengths=[80_000, 10_000, 10_000],
        shuffling=True,
    )
    trainer = pl.Trainer(max_epochs=2)
    test_input = torch.normal(0, 1, size=(config["batch_size"], state_dimension))
    forw = model.forward(test_input)
    mlflow.models.signature

    #     mats = model.construct_full_matrix(forw)
    #     print(f"{mats.shape=}")
    #     print(f"{mats}")

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a surrogate of the inverse of the Gauss-Newton matrix"
    )
    parser.add_argument("dim", type=int, help="state dimension")
    parser.add_argument("training_data", type=str, help="training data")
    args = parser.parse_args()
    main(state_dimension=args.dim, data_path=args.training_data)
