import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional

import torch
import random
import numpy as np
from safetensors.torch import load_file

import data.era5__diffusion_dataset as data
from torch.utils.data import DataLoader

import model.diffusion.model as model
import model.schedulers as schedulers
import model.optimizers as optimizers
import model.utility as model_utility
import model.objectives.diffusion_loss as loss
import model.diffusion.train as training

import utils.wandb_helper as wbhelp
import utils.utility as utility

from accelerate import Accelerator


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # reused variables
    save_path = cfg['experiment']['save_path']

    # get accelerator
    accelerator = Accelerator(gradient_accumulation_steps=cfg['distributed_training']['grad_accumulate'])
    
    # intial setup
    seed = cfg.get("training.seed", 42)  # Default to 42 if not specified
    utility.set_random_seeds(seed)
    
    # logging/checkpointing setup
    if accelerator.is_main_process:
        utility.validate_and_create_save_path(cfg['experiment']['save_path'], cfg['experiment']['experiment_name'])    
        wbhelp.init_wandb(
            project_name=cfg['experiment']['project_name'],
            run_name=cfg['experiment']['experiment_name'],
            config_class=cfg,
            save_path=save_path
        )
    
    # get the diffusion dataset
    train_dataset = data.ERA5DiffusionDataset(
        root_dir=cfg['dataset']['root_dir'],
        start_date=cfg['training']['dataset']['start_date'],
        end_date=cfg['training']['dataset']['end_date'],
        timesteps=cfg['dataset']['timesteps'],
        cfg=cfg,
    )
    valid_dataset = data.ERA5DiffusionDataset(
        root_dir=cfg['dataset']['root_dir'],
        start_date=cfg['training']['validation_dataset']['start_date'],
        end_date=cfg['training']['validation_dataset']['end_date'],
        timesteps=cfg['dataset']['timesteps'],
        cfg=cfg,
    )
        
    # returns sample (85x32x64), forcings+constants (11x32x64), noise (85x32x64), random timestep (1)
    sample_x, fc, _, _ = train_dataset[0]
    
    channels, domain_x, domain_y = sample_x.shape
    fc_channels, _, _ = fc.shape
    
    # get dataloader
    train_dl = DataLoader(
        train_dataset, batch_size=cfg['distributed_training']['total_batch_size'], 
        shuffle=True, num_workers=cfg['distributed_training']['workers'], 
        drop_last=True, pin_memory=True, persistent_workers=True, multiprocessing_context="spawn"
    )
    valid_dl = DataLoader(
        valid_dataset, batch_size=cfg['distributed_training']['total_batch_size'], 
        shuffle=True, num_workers=cfg['distributed_training']['workers'], 
        drop_last=True, pin_memory=True, persistent_workers=True, multiprocessing_context="spawn"
    )    

    # get diffusion model
    diffusion_model = model.get_unet_based_model(
        x = domain_x, y = domain_y,
        channels = channels, cfg = cfg
    )
    if accelerator.is_main_process:
        wbhelp.save_model_architecture(diffusion_model, cfg['experiment']['save_path'])

    # load from checkpoint if needed
    if cfg['experiment']['from_checkpoint'] is not None:
        model_utility.load_model_weights(diffusion_model, cfg['experiment']['from_checkpoint'])
    
    # get determinsitic model
    pred_model = model.get_unet_based_model(
        domain_x, domain_y, channels+fc_channels, 
        channels, cfg 
    )
    
    # deterministic force load from checkpoint
    assert cfg['deterministc']['from_checkpoint'] is not None, 'Deterministic checkpoint is required!'
    model_utility.load_model_weights(pred_model, cfg['deterministc']['from_checkpoint'])
    
    # get optimizer
    optimizer = optimizers.get_adamw(diffusion_model, cfg['optimization']['lr'])
    
    # get schedular
    scheduler = schedulers.get_onecycle_lr(
        optimizer, cfg['optimization']['max_lr'],
        cfg['training_info']['epochs'], len(train_dl)
    )
    
    # load from state if needed
    if cfg['experiment']['from_state'] is not None:
        model_utility.load_training_state(
            accelerator, cfg['experiment']['from_state'], 
            diffusion_model, optimizer, scheduler
        )
    
    # prepare for distributed training
    train_dl, valid_dl, diffusion_model, pred_model, optimizer, scheduler = accelerator.prepare(
        train_dl, valid_dl, diffusion_model, pred_model, optimizer, scheduler
    )
    
    # get criterion
    criterion = loss.get_diffusion_loss()

    # distribute training
    training.training_loop(
        accelerator = accelerator, 
        train = train_dl, 
        vali = valid_dl, 
        model = diffusion_model, 
        operator = pred_model, 
        epochs = cfg['training_info']['epochs'], 
        criterion = criterion, 
        save_path = cfg['experiment']['save_path'], 
        optimizer = optimizer, 
        scheduler = scheduler, 
        t_timesteps: cfg['training_info']['timesteps'], 
        val_delay = cfg['training_info']['validation_delay'], 
        loading_bar = True,
        config = cfg
    )
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()