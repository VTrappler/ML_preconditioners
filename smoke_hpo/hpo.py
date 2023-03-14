import sys

sys.path.append("..")
from prec_data.data import TangentLinearDataModule
from prec_models.models_unstructured import BandMatrix, LowRank
from prec_models.models_spectral import SVDConvolutional
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
import functools
from pytorch_lightning.loggers import TensorBoardLogger
import mlflow
from ray.tune.integration.mlflow import mlflow_mixin

from ray import air, tune
from ray.air.integrations.mlflow import setup_mlflow
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from omegaconf import OmegaConf
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")


def construct_model_class(cl, **kwargs):
    import functools

    class dummy(cl):
        __init__ = functools.partialmethod(cl.__init__, **kwargs)

    return dummy

@mlflow_mixin
def trainable(config):
    setup_mlflow(config=config, experiment_name=config["mlflow"]["experiment_name"])
    metrics = {"loss": "Loss/val_loss"}
    mlflow.pytorch.autolog()
    state_dimension = config["data"]["dimension"]
    data_path = config["data"]["data_path"]

    torch_model = construct_model_class(LowRank, rank=config["architecture"]["rank"])
    mlflow.log_params(config["architecture"])
    mlflow.log_params(config["data"])
    # mlflow.log_params(config["optimizer"])

    model = torch_model(state_dimension=state_dimension, config=config["architecture"])
    datamodule = TangentLinearDataModule(
        path=data_path,
        batch_size=config["architecture"]["batch_size"],
        num_workers=4,
        splitting_lengths=[0.8, 0.1, 0.1],
        shuffling=True,
    )
    trainer = pl.Trainer(
        max_epochs=config["optimizer"]["epochs"],
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
        enable_progress_bar=False,
    )

    test_input = torch.normal(
        0, 1, size=(config["architecture"]["batch_size"], state_dimension)
    )
    forw = model.forward(test_input)
    print(f"{forw.shape=}")

    #     mats = model.construct_full_matrix(forw)
    #     print(f"{mats.shape=}")
    #     print(f"{mats}")
    signature = mlflow.models.signature.infer_signature(
        test_input.detach().numpy(), forw.detach().numpy()
    )
    trainer.fit(
        model,
        datamodule,
    )
        # mlflow.pytorch.log_model(model, "smoke_model", signature=signature)


def tuner(config):
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    # config = {
    #     "data": {
    #         "dimension": 20,
    #         "data_path": "/home/raw_data/data.pkl",
    #     },
    #     "architecture": {
    #         "rank": 5,
    #         "n_layers": tune.choice([2, 3, 4]),
    #         "neurons_per_layer": tune.choice([32, 64]),
    #         "batch_size": tune.choice([32, 64, 128]),
    #     },
    #     "optimizer": {
    #         "lr": tune.loguniform(1e-4, 1e-1),
    #         "epochs": 50,
    #     },
    #     "mlflow": {
    #         "experiment_name": experiment_name,
    #         "tracking_uri": mlflow.get_tracking_uri(),
    #     },
    # }
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=config["tuner"]["num_samples"],
            max_concurrent_trials=config["tuner"]["max_concurrent_trials"],
        ),
        param_space=config,
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPO")
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    if OmegaConf._get_resolver("choice") is None:
        OmegaConf.register_new_resolver(
            "choice", lambda *numbers: tune.choice([int(n) for n in numbers])
        )
    if OmegaConf._get_resolver("loguniform") is None:
        OmegaConf.register_new_resolver(
            "loguniform",
            lambda lower, upper: tune.loguniform(float(lower), float(upper)),
        )
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    tuner(config)
