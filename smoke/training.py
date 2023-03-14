import sys
sys.path.append('..')
import os
from prec_data.data import TangentLinearDataModule
from prec_models.models_spectral import SVDPrec, SVDConvolutional
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import argparse
import functools
from pytorch_lightning.loggers import CSVLogger
import mlflow
from omegaconf import OmegaConf
from os.path import join, dirname
# from dotenv import load_dotenv, dotenv_values
def construct_model_class(cl, **kwargs):
    class dummy(cl):
        __init__ = functools.partialmethod(cl.__init__, **kwargs)

    return dummy




def main(config):
    mlflow.pytorch.autolog(log_models=False) # Logging model with signature at the end instead
    state_dimension = config['data']['dimension']
    print(f"{state_dimension=}")
    data_path = config['data']['data_path']

    torch_model = construct_model_class(SVDConvolutional, rank=config['architecture']['rank'])
    mlflow.log_params(config['architecture'])
    mlflow.log_params(config['data'])
    mlflow.log_params(config['optimizer'])


    model = torch_model(state_dimension=state_dimension, config=config['architecture'])
    datamodule = TangentLinearDataModule(
        path=data_path,
        batch_size=config['architecture']["batch_size"],
        num_workers=4,
        splitting_lengths=[0.8, 0.1, 0.1],
        shuffling=True,
    )
    trainer = pl.Trainer(max_epochs=config['optimizer']['epochs'], logger=CSVLogger('/home/smoke/'))
    test_input = torch.normal(0, 1, size=(config['architecture']["batch_size"], state_dimension))
    forw = model.forward(test_input)    
    print(f"{forw.shape=}")


    #     mats = model.construct_full_matrix(forw)
    #     print(f"{mats.shape=}")
    #     print(f"{mats}")
    trainer.fit(model, datamodule)
    print(trainer.logged_metrics)
    with open("/home/smoke/metrics.yaml", 'w') as fp:
        metrics_dict = {k: float(v) for k, v in trainer.logged_metrics.items()}
        OmegaConf.save(config=metrics_dict, f=fp)
    signature = mlflow.models.signature.infer_signature(test_input.detach().numpy(), forw.detach().numpy())
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