import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional

import torch
import random
import numpy as np
from safetensors.torch import load_file

import data.era5_dataset as data
from torch.utils.data import DataLoader

import model.deterministic.model as model
import model.schedulers as schedulers
import model.optimizers as optimizers
import model.utility as model_utility
import model.objectives.operator_loss as loss
import model.deterministic.train as training

import utils.wandb_helper as wbhelp
import utils.utility as utility

from accelerate import Accelerator

def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # reused variables
    save_path = cfg['experiment']['save_path']
    
    # intial setup
    seed = cfg.get("training.seed", 42)  # Default to 42 if not specified
    set_random_seeds(seed)
    
    # logging/checkpointing setup
    # utility.validate_and_create_save_path(cfg['experiment']['save_path'], cfg['experiment']['experiment_name'])    
    wbhelp.init_wandb(
        project_name=cfg['experiment']['project_name'],
        run_name=cfg['experiment']['experiment_name'],
        config_class=cfg,
        save_path=save_path
    )
    
    # get datasets
    train_dataset = data.ERA5Dataset(
        root_dir=cfg['dataset']['root_dir'],
        start_date=cfg['training']['dataset']['start_date'],
        end_date=cfg['training']['dataset']['end_date'],
        forecast_steps=cfg['dataset']['forecast_steps'],
        cfg=cfg,
    )
    valid_dataset = data.ERA5Dataset(
        root_dir=cfg['dataset']['root_dir'],
        start_date=cfg['training']['validation_dataset']['start_date'],
        end_date=cfg['training']['validation_dataset']['end_date'],
        forecast_steps=cfg['dataset']['forecast_steps'],
        cfg=cfg,
    )
    
    sample_x, sample_y, _ = train_dataset[0]
    
    in_c, domain_x, domain_y = sample_x.shape
    out_c, _, _ = sample_y.shape
    
    # get dataloader
    train_dl = DataLoader(
        train_dataset, batch_size=cfg['distributed_training']['total_batch_size'], shuffle=True, 
        num_workers=cfg['distributed_training']['workers'], drop_last=True, pin_memory=True
    )
    valid_dl = DataLoader(
        valid_dataset, batch_size=cfg['distributed_training']['total_batch_size'], shuffle=True, 
        num_workers=cfg['distributed_training']['workers'], drop_last=True, pin_memory=True
    )    
    
    # get model
    pred_model = model.get_unet_based_model(
        domain_x, domain_y, in_c, out_c, cfg
    )
    wbhelp.save_model_architecture(pred_model, cfg['experiment']['save_path'])
    
    # load from checkpoint if needed
    if cfg['experiment']['from_checkpoint'] is not None:
        state_dict = load_file(cfg['experiment']['from_checkpoint'])
        model.load_model_weights(pred_model, state_dict)
    
    # get optimizer
    optimizer = optimizers.get_adamw(pred_model, cfg['optimization']['lr'])
    
    # get schedular
    scheduler = schedulers.get_onecycle_lr(
        optimizer, cfg['optimization']['max_lr'],
        cfg['training_info']['epochs'], len(train_dl)
    )
    
    # get accelerator
    accelerator = Accelerator(gradient_accumulation_steps=cfg['distributed_training']['grad_accumulate'])
    
    # load from state if needed
    if cfg['experiment']['from_state'] is not None:
        model_utility.load_training_state(
            accelerator, cfg['experiment']['from_state'], 
            pred_model, optimizer, scheduler
        )
    
    # prepare for distributed training
    train_dl, valid_dl, pred_model, optimizer, scheduler = accelerator.prepare(
        train_dl, valid_dl, pred_model, optimizer, scheduler
    )
    
    # get criterion
    criterion = loss.OperatorLoss(
        cfg['loss']['mse'], 
        cfg['loss']['mae']
    )
    
    # distributive training
    training.training_loop(
        accelerator = accelerator, 
        train = train_dl, 
        valid = valid_dl, 
        model = pred_model, 
        epochs = cfg['training_info']['epochs'], 
        patience = cfg['training_info']['patience'], 
        criterion = criterion, 
        save_path = save_path, 
        optimizer = optimizer, 
        scheduler = scheduler,
        loading_bar = True
    )
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()