from prec_data.data import TangentLinearDataModule
from prec_models.models_unstructured import BandMatrix, LowRank
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
import functools
from pytorch_lightning.loggers import TensorBoardLogger
import mlflow

from ray import air, tune
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.integration.pytorch_lightning import TuneReportCallback


from omegaconf import OmegaConf


def construct_model_class(cl, **kwargs):
    import functools
    class dummy(cl):
        __init__ = functools.partialmethod(cl.__init__, **kwargs)

    return dummy


def trainable(config):
    setup_mlflow(config)
    metrics = {"loss": "Loss/val_loss"}
    mlflow.pytorch.autolog()
    state_dimension = config["data"]["dimension"]
    print(f"{state_dimension=}")
    data_path = config["data"]["data_path"]

    torch_model = construct_model_class(LowRank, rank=config["architecture"]["rank"])
    mlflow.log_params(config["architecture"])
    mlflow.log_params(config["data"])
    mlflow.log_params(config["optimizer"])

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
    trainer.fit(
        model,
        datamodule,
    )
    signature = mlflow.models.signature.infer_signature(
        test_input.detach().numpy(), forw.detach().numpy()
    )
    # mlflow.pytorch.log_model(model, "smoke_model", signature=signature)


def tuner(experiment_name, num_samples=10):
    mlflow.set_experiment(experiment_name)
    config = {
        "data": {
            "dimension": 20,
            "data_path": "/home/raw_data/data.pkl",
        },
        "architecture": {
            "rank": 5,
            "n_layers": tune.choice([2, 3, 4]),
            "neurons_per_layer": tune.choice([32, 64]),
            "batch_size": tune.choice([32, 64, 128]),
        },
        "optimizer": {
            "lr": tune.loguniform(1e-4, 1e-1),
            "epochs": 50,
        },
        "mlflow": {
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
        },
    }
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_samples,
            max_concurrent_trials=2    
        ),
        run_config=air.RunConfig(
            name="tune_smoke",
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    tuner(num_samples=10, experiment_name="smoke_hpo_pl_mlflow_3")
