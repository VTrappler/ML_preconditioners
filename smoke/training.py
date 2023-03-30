import shutil
import sys

sys.path.append("..")
import argparse
import os

import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from prec_data.data import TangentLinearDataModule
from prec_models import construct_model_class
from prec_models.models_spectral import SVDConvolutional, SVDPrec, DeflationPrec
from prec_models.models_limitedmemoryprec import LMPPrec, LimitedMemoryPrecRegularized
from prec_models.models_unstructured import LowRank

logs_path = os.path.join(os.sep, "root", "log_dump", "smoke")
smoke_path = os.path.join(os.sep, "home", "smoke")


# from dotenv import load_dotenv, dotenv_values
model_list = [
    SVDConvolutional,
    SVDPrec,
    DeflationPrec,
    LMPPrec,
    LimitedMemoryPrecRegularized,
    LowRank,
]
model_classes = {cl.__name__: cl for cl in model_list}


def main(config):
    mlflow.pytorch.autolog(
        log_models=False
    )  # Logging model with signature at the end instead
    state_dimension = config["model"]["dimension"]
    print(f"{state_dimension=}")
    data_path = config["data"]["data_path"]

    torch_model = construct_model_class(
        model_classes[config["architecture"]["class"]],
        rank=config["architecture"]["rank"],
    )
    model = torch_model(state_dimension=state_dimension, config=config["architecture"])

    mlflow.log_params(config["architecture"])
    mlflow.log_params(config["data"])
    config["optimizer"].pop("lr", None)
    mlflow.log_params(config["optimizer"])

    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    with open(os.path.join(smoke_path, "mlflow_run_id.yaml"), "w") as fh:
        run_id_dict = {"run_id": run.info.run_id, "data_path": data_path}
        OmegaConf.save(config=run_id_dict, f=fh)

    datamodule = TangentLinearDataModule(
        path=data_path,
        batch_size=config["architecture"]["batch_size"],
        num_workers=4,
        splitting_lengths=[0.8, 0.1, 0.1],
        shuffling=True,
        normalization=False,
    )
    trainer = pl.Trainer(
        max_epochs=config["optimizer"]["epochs"],
        logger=CSVLogger(logs_path, version="smoke"),
    )
    test_input = torch.normal(
        0, 1, size=(config["architecture"]["batch_size"], state_dimension)
    )
    forw = model.forward(test_input)
    print(f"{forw.shape=}")

    #     mats = model.construct_full_matrix(forw)
    #     print(f"{mats.shape=}")
    #     print(f"{mats}")
    trainer.fit(model, datamodule)
    print(trainer.logged_metrics)
    shutil.copyfile(
        os.path.join(logs_path, "lightning_logs", "smoke", "metrics.csv"), "./logs.csv"
    )

    with open(os.path.join(smoke_path, "metrics.yaml"), "w") as fp:
        metrics_dict = {k: float(v) for k, v in trainer.logged_metrics.items()}
        # metrics_dict["run_id"] = run.info.run_id
        OmegaConf.save(config=metrics_dict, f=fp)

    signature = mlflow.models.signature.infer_signature(
        test_input.detach().numpy(), forw.detach().numpy()
    )
    mlflow.pytorch.log_model(model, "smoke_model", signature=signature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a surrogate of the inverse of the Gauss-Newton matrix"
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp-name", type=str, default="expname")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    mlflow.set_experiment(args.exp_name)
    print(mlflow.get_tracking_uri())
    print(os.getcwd())
    main(conf)
