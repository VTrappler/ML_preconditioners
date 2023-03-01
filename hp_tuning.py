from data import CholeskyDataModule
from models import CholeskyRegularized, Cholesky, LowRank, BandMatrix

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import argparse
import os
import functools
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.suggest.hyperopt import HyperOptSearch
import randomname
import yaml

def LowRank_k(cls, rank):
    class low_rank(cls):
        __init__ = functools.partialmethod(cls.__init__, rank=rank)

    return low_rank


def BandMatrix_k(cls, bw):
    class bandmat(cls):
        __init__ = functools.partialmethod(cls.__init__, bw=bw)

    return bandmat

torch_model = LowRank_k(LowRank, rank=5)
torch_model = BandMatrix_k(BandMatrix, bw=2)


def train_tune(config, state_dimension, data_path, num_epochs):
    model = torch_model(state_dimension=state_dimension, config=config)
    datamodule = CholeskyDataModule(path=data_path,
                                    batch_size=config['batch_size'],
                                    num_workers=4,
                                    splitting_lengths = [80_000, 10_000, 10_000],
                                    shuffling=True)
    # print(config)
    trainer = pl.Trainer(max_epochs=num_epochs,
                         callbacks=TuneReportCheckpointCallback(metrics={"loss": "Loss/val_loss"},
                                                                filename="checkpoint",
                                                                on="validation_end"),
                         logger=TensorBoardLogger(save_dir="", version="."),
                         enable_progress_bar=False)
    trainer.fit(model, datamodule)
    return model




    
def train_tune_asha(state_dimension, data_path, dir='ray', num_samples=50, num_epochs=200, gpus_per_trial=0):
    config = {
              "layers": [
                         tune.choice([state_dimension]),
                         tune.choice([2**i for i in range(5, 6)]),
                         tune.choice([2**i for i in range(6, 8)]),
                         tune.choice([2**i for i in range(7, 8)]),
                         tune.choice([2**i for i in range(6, 8)])
                        ],
              "lr": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
              "n_layers_diag": tune.choice([2, 3]),
              "n_layers_offdiag": tune.choice([2, 3]),
              "batch_size": tune.choice([16, 32, 64]),
             }

    if not os.path.exists(f"/da_dev/GNlearning/{dir}/"):
        os.makedirs(f"/da_dev/GNlearning/{dir}/")
  
    _existing_xps = os.listdir(f"/da_dev/GNlearning/{dir}/")

# Generate experiment name
    _randomize_name = True
    while _randomize_name:
        name = randomname.get_name()
        if name not in _existing_xps:
            break
    
    name = randomname.get_name()
    print(f"\n\n\n\nName: {name}\n\n\n\n\n")
    
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=5,
                              reduction_factor=1.1)
    
    reporter = CLIReporter(parameter_columns=["layers", "n_layers_offdiag", "n_layers_diag", "lr", "batch_size"],
                           metric_columns=["loss", "training_iteration"])
    
    train_tune_with_parameters = tune.with_parameters(train_tune,
                                                      state_dimension=state_dimension,
                                                      data_path=data_path,
                                                      num_epochs=num_epochs)
    
    hyperopt_search = HyperOptSearch(metric="loss", mode="min", n_initial_points=2)
    
    with open(f"/da_dev/GNlearning/{name}_log.txt", 'w') as f:
        f.write(name)
        
    
    
    analysis = tune.run(train_tune_with_parameters,
                        metric="loss",
                        mode="min",
                        config=config,
                        search_alg=hyperopt_search,
                        num_samples=num_samples,
                        scheduler=scheduler,
                        name=name,
                        checkpoint_at_end=True,
                        keep_checkpoints_num=2,
                        local_dir = f"{dir}/",
                        max_concurrent_trials=10)

    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    
    print(f"{analysis.best_trial=}")  # Get best trial
    print(f"{analysis.best_config=}")  # Get best trial's hyperparameters
    print(f"{analysis.best_logdir=}")  # Get best trial's logdir
    print(f"{analysis.best_checkpoint=}")  # Get best trial's best checkpoint
    print(f"{analysis.best_result=}")  # Get best trial's last results
    print(f"{analysis.best_result_df=}")  # Get best result as pandas dataframe
    
    best_trained_model = torch_model(state_dimension=state_dimension, config=best_trial.config)
    best_checkpoint_dir = best_trial.checkpoint.value
    with open(f"/da_dev/GNlearning/{name}_log.txt", 'w') as f:
        f.write(best_checkpoint_dir)
        f.write("Best trial config: {}".format(best_trial.config))
        f.write(f"{analysis.best_trial=}")  # Get best trial
        f.write(f"{analysis.best_config=}")  # Get best trial's hyperparameters
        f.write(f"{analysis.best_logdir=}")  # Get best trial's logdir
        f.write(f"{analysis.best_checkpoint=}")  # Get best trial's best checkpoint
        
    load_ckpt = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(load_ckpt["state_dict"])
    torch.save({'model_state_dict': best_trained_model.state_dict(),
                # 'optimizer_state_dict': best_trained_model.optimizers().state_dict(),
                'config': best_trial.config},
               f"/da_dev/GNlearning/{dir}/{name}/best_{name}.pth")
    print(f"Best model saved in /da_dev/GNlearning/{dir}/{name}/{name}_best.pth")
    with open(f"/da_dev/GNlearning/{dir}/{name}/{name}_config.yaml", 'w') as yaml_file:
        yaml.dump(best_trial.config, yaml_file, default_flow_style=False)
    
    
if __name__ == '__main__':
    # train_tune_asha(state_dimension=3, data_path="/da_dev/GNlearning/data/training_affine_3_100000_20obs.pkl", dir='affine')
    # train_tune_asha(state_dimension=3, data_path="/da_dev/GNlearning/data/training_ns_3_100000_20obs.pkl", dir='nonsmooth')
    # train_tune_asha(state_dimension=6, data_path="/da_dev/GNlearning/data/training_ns_6_100000_20obs.pkl", dir='nonsmooth')
    train_tune_asha(state_dimension=10, data_path="/da_dev/GNlearning/data/training_id_10_100000_10obs.pkl", dir='band')
    # train_tune_asha(state_dimension=6, data_path="/da_dev/GNlearning/data/training_6_100000_20obs.pkl")
    # train_tune_asha(state_dimension=10, data_path="/da_dev/GNlearning/data/training_10_100000_20obs.pkl")
    # train_tune_asha(state_dimension=20, data_path="/da_dev/GNlearning/data/training_20_100000_20obs.pkl", dir='ray')