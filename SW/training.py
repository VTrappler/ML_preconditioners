import shutil
import sys

sys.path.append("..")
import argparse
import os

import lightning.pytorch as pl
import mlflow
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger

from omegaconf import OmegaConf
from prec_data.data_sw import SWDataModule
from prec_models import construct_model_class
from prec_models.models_limitedmemoryprec import LimitedMemoryPrecRegularized, LMPPrec
from prec_models.models_spectral import (
    DeflationPrec,
    SVDConvolutional,
    SVDConvolutionalSPAI,
    SVDPrec,
)
from prec_models.sw_unet import SW_UNet, SW_Conv, SW_TransUNet

model_list = [SW_UNet, SW_Conv, SW_TransUNet]
model_classes = {cl.__name__: cl for cl in model_list}
# create your own theme!


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
logs_path = os.path.join(os.sep, "data", "data_data_assimilation", "log_dump", "SW")
exp_path = os.path.join(os.sep, "GNlearning", "SW")

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


def main(config):
    loggers = []
    if config["misc"]["mlflow"]:
        mlflow.set_experiment("SW_GN_learning")
        mlflow.pytorch.autolog(
            log_models=False,
            # log_model_signatures=True,
            # log_input_examples=True,
            # log_datasets=True,
        )  # Logging model with signature at the end instead
        mlflow.start_run()
        run = mlflow.active_run()
        print(f"{run=}")
        print("Active run_id: {}".format(run.info.run_id))
    else:
        print("No MLFLOW -> Tensorboard logger")
        tb_logger = TensorBoardLogger(save_dir=logs_path, name="SW")
        loggers.append(tb_logger)

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

    if config["misc"]["mlflow"]:
        mlflow.log_params(flatten_dict(config))
        tmp_storage_dir = os.path.join(
            os.sep,
            "data",
            "data_data_assimilation",
            "shallow_water",
            "tmp_model_storage",
            run.info.run_id,
        )
        os.makedirs(
            tmp_storage_dir,
            exist_ok=True,
        )
        shutil.copyfile(
            os.path.join(exp_path, "config.yaml"),
            os.path.join(tmp_storage_dir, "config.yaml"),
        )
        with open(os.path.join(artifacts_path, "mlflow_run_id.yaml"), "w") as fh:
            run_id_dict = {
                "run_id": run.info.run_id,
                "data_path": data_path,
                "data_folder": data_folder,
                "model_path": os.path.join(tmp_storage_dir, "model.pth"),
            }
            OmegaConf.save(config=run_id_dict, f=fh)

        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(
                mlflow.active_run().info.experiment_id
            ).name,
            run_id=run.info.run_id,
        )
        loggers.append(mlf_logger)

    datamodule = SWDataModule(  # TODO: change
        folder=config["data"]["data_folder"],
        # nsamples=config["data"]["nsamples"],
        # dim=state_dimension,
        batch_size=config["architecture"]["batch_size"],
        num_workers=5,
        splitting_lengths=[0.9, 0.1, 0.0],
        shuffling=True,
        # normalization=False,
    )
    datamodule.setup(None)
    loggers.append(CSVLogger(logs_path, version="SW"))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["optimizer"]["epochs"],
        logger=loggers,
        callbacks=[progress_bar],
        enable_checkpointing=False,
    )
    test_input = torch.normal(
        0, 1, size=(config["architecture"]["batch_size"], state_dimension)
    )
    model = model.to(device)
    test_output = model.forward(test_input.to(device))
    # print(f"{forw.shape=}")
    signature = mlflow.models.signature.infer_signature(
        test_input.cpu().detach().numpy(), test_output.cpu().detach().numpy()
    )
    print(signature)

    trainer.fit(
        model,
        datamodule=datamodule,
    )
    print(trainer.logged_metrics)

    if config["misc"]["mlflow"]:
        shutil.copyfile(
            os.path.join(logs_path, "lightning_logs", "SW", "metrics.csv"),
            os.path.join(artifacts_path, "training_logs.csv"),
        )

        with open(os.path.join(artifacts_path, "metrics.yaml"), "w") as fp:
            metrics_dict = {k: float(v) for k, v in trainer.logged_metrics.items()}
            metrics_dict["run_id"] = run.info.run_id
            OmegaConf.save(config=metrics_dict, f=fp)

        shutil.copyfile(
            os.path.join(artifacts_path, "training_logs.csv"),
            os.path.join(tmp_storage_dir, "training_logs.csv"),
        )

        shutil.copyfile(
            os.path.join(artifacts_path, "metrics.yaml"),
            os.path.join(tmp_storage_dir, "metrics.yaml"),
        )

        # mlflow.pytorch.log_model(
        #     artifact_path="SW_model",
        #     # registered_model_name="SW_model",
        #     signature=signature,
        # )

        torch.save(
            model.state_dict(),
            os.path.join(tmp_storage_dir, "model.pth"),
        )
        print(tmp_storage_dir)


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
