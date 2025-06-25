import glob
import logging
import multiprocessing as mp
import os

import certifi
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

import wandb
from data import get_dataloaders
from src.fast_genetics.utils.train_fns import consistency_check, flatten_dict
from src.models.ldsr_trainer import LDSRTrainer
from src.models.nets import get_model_setup
from src.models.trainer import (
    WASP,
    SNPTrainerChol,
    WASP_new,
)

torch.set_num_threads(1)

os.environ["SSL_CERT_FILE"] = certifi.where()

logging.disable(logging.INFO)


def num_downsampler(window_size, target_length):
    return int(np.log2(window_size / target_length))


OmegaConf.register_new_resolver("num_downsampler", num_downsampler)


def get_trainer(name):
    CLS = {
        "ldsr": LDSRTrainer,
        "mle_chol": SNPTrainerChol,
        "wasp": WASP,
        "wasp_new": WASP_new,
    }
    return CLS[name]


@rank_zero_only
def init_wandb(cfg):
    wandb.init()
    wandb.config.update(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))


@rank_zero_only
def update_wandb(cfg):
    wandb.config.update(cfg)


@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    if cfg.wandb.use:
        # wandb.init()
        # wandb.config.update(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
        init_wandb(cfg)

    consistency_check(cfg)
    torch.set_float32_matmul_precision("high")

    # ### Load data
    pl.seed_everything(cfg.model.seed, workers=True)
    print("Getting dataloaders.")
    train_dataloader, test_dataloader, geno_track_names, anno_track_names = get_dataloaders(cfg.data)

    # ### Setup x0_model
    print("Setting up model.")
    model_class, nn_params = get_model_setup(cfg.architecture, geno_track_names, anno_track_names)  # TODO

    print(cfg)
    # ### Pick model
    trainer = get_trainer(cfg.model.loss)
    if not cfg.model.restart:
        model = trainer(
            model_class,
            total_steps=len(train_dataloader) * cfg.train.n_epoch,
            scheduler_name=cfg.train.scheduler,
            n_pop=cfg.data.N,
            n_snp=cfg.model.M,
            include_D=cfg.model.include_D,
            eps=cfg.model.eps,
            nn_params=nn_params,
            geno_track_names=geno_track_names,
            anno_track_names=anno_track_names,
            beta_eval_thresh=cfg.model.beta_eval_thresh,
            seed=cfg.model.seed,
            wandb=cfg.wandb.use,
            **OmegaConf.to_container(cfg.train, resolve=True),
        )
        ckpt_path = None
    else:
        ckpt_path = f"checkpoints/{cfg.model.restart}"
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, "*.ckpt")), key=os.path.getmtime)
        print(f"Restarting model | {ckpt_path=}")
        model = trainer.load_from_checkpoint(ckpt_path)

    # ### Preconfigure
    params_n = sum(par.numel() for par in model.parameters() if par.requires_grad)
    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"{params_n / 1e6:2.2f}M params | {gpu_type}")

    # ### Train
    if cfg.wandb.use:
        logger = WandbLogger(project=cfg.wandb.project)
        output_dir = logger.experiment.name
        update_wandb({"params": params_n, "params(M)": params_n / 1e6, "gpu_type": gpu_type})
        # wandb.config.update({"params": params_n, "params(M)": params_n / 1e6, "gpu_type": gpu_type})
    else:
        logger = None
        output_dir = "run"
    lightning_model = model

    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=f"checkpoints/{output_dir}", save_on_train_epoch_end=False))

    trainer = Trainer(
        max_epochs=cfg.train.n_epoch,
        accelerator="auto",
        devices=max([1, torch.cuda.device_count()]),
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        accumulate_grad_batches=cfg.train.accumulate,
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)

    if cfg.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    train()
