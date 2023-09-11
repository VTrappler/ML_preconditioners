import shutil
import sys

sys.path.append("..")
import argparse
import os

import mlflow
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

# create your own theme!


from omegaconf import OmegaConf
from lightning.pytorch.loggers import CSVLogger

from prec_data.data_memmap import TangentLinearDataModuleMEMMAP
from prec_models import construct_model_class
from prec_models.models_limitedmemoryprec import LimitedMemoryPrecRegularized, LMPPrec
from prec_models.models_spectral import (
    DeflationPrec,
    SVDConvolutional,
    SVDPrec,
    SVDConvolutionalSPAI,
)
from prec_models.models_unstructured import LowRank

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    )
)


# logs_path = os.path.join(os.sep, "root", "log_dump", "smoke")
# exp_path = os.path.join(os.sep, "home", "smoke")
logs_path = os.path.join(os.sep, "data", "data_data_assimilation", "log_dump", "lorenz")
exp_path = os.path.join(os.sep, "GNlearning", "lorenz")

artifacts_path = os.path.join(exp_path, "artifacts")

from collections.abc import MutableMapping


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# from dotenv import load_dotenv, dotenv_values
model_list = [
    SVDConvolutional,
    SVDPrec,
    DeflationPrec,
    LMPPrec,
    LimitedMemoryPrecRegularized,
    LowRank,
    SVDConvolutionalSPAI,
]
model_classes = {cl.__name__: cl for cl in model_list}


def main(config):
    mlflow.set_experiment("smoke_train_tr")
    mlflow.pytorch.autolog(
        log_models=False
    )  # Logging model with signature at the end instead
    mlflow.start_run()
    run = mlflow.active_run()
    print(f"{run=}")
    print("Active run_id: {}".format(run.info.run_id))
    mlf_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(
            mlflow.active_run().info.experiment_id
        ).name,
        run_id=run.info.run_id,
    )

    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.device_count()=}")
    print(f"{torch.cuda.current_device()=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    state_dimension = config["model"]["dimension"]
    print(f"{state_dimension=}")

    data_path = config["data"].get("data_path", None)
    data_folder = config["data"].get("data_folder", None)

    torch_model = construct_model_class(
        model_classes[config["architecture"]["class"]],
        rank=config["architecture"]["rank"],
    )
    model = torch_model(state_dimension=state_dimension, config=config["architecture"])

    config["optimizer"].pop("lr", None)
    mlflow.log_params(flatten_dict(config))

    with open(os.path.join(artifacts_path, "mlflow_run_id.yaml"), "w") as fh:
        run_id_dict = {
            "run_id": run.info.run_id,
            "data_path": data_path,
            "data_folder": data_folder,
        }
        OmegaConf.save(config=run_id_dict, f=fh)

    datamodule = TangentLinearDataModuleMEMMAP(
        path=config["data"]["data_folder"],
        nsamples=config["data"]["nsamples"],
        dim=state_dimension,
        window=config["model"]["window"],
        batch_size=config["architecture"]["batch_size"],
        num_workers=5,
        splitting_lengths=[0.1, 0.1, 0.8],
        shuffling=True,
        normalization=False,
    )
    datamodule.setup(None)

    trainer = pl.Trainer(
        accelerator="cpu",
        # devices=1,
        max_epochs=config["optimizer"]["epochs"],
        logger=[mlf_logger, CSVLogger(logs_path, version="smoke")],
        callbacks=[progress_bar],
        enable_checkpointing=False,
    )
    test_input = torch.normal(
        0, 1, size=(config["architecture"]["batch_size"], state_dimension)
    )

    forw = model.forward(test_input)
    print(f"{forw.shape=}")
    # print(f"{forw=}")

    # mats = model.construct_full_matrix(forw)
    # print(f"{mats.shape=}")
    # print(f"{mats}")

    trainer.fit(
        model,
        datamodule=datamodule,
    )
    print(trainer.logged_metrics)
    shutil.copyfile(
        os.path.join(logs_path, "lightning_logs", "smoke", "metrics.csv"),
        os.path.join(artifacts_path, "training_logs.csv"),
    )

    with open(os.path.join(artifacts_path, "metrics.yaml"), "w") as fp:
        metrics_dict = {k: float(v) for k, v in trainer.logged_metrics.items()}
        metrics_dict["run_id"] = run.info.run_id
        OmegaConf.save(config=metrics_dict, f=fp)

    signature = mlflow.models.signature.infer_signature(
        test_input.detach().numpy(), forw.detach().numpy()
    )
    mlflow.pytorch.log_model(
        model, "smoke_model", signature=signature, code_paths=["."]
    )


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