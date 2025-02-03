import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional

import torch
import random
import os
import numpy as np
from safetensors.torch import load_file

import data.era5_dataset as data

import model.deterministic.model as model
import model.utility as model_utility

import evaluation.utility as evaluation

import utils.utility as utility


@hydra.main(version_base=None, config_path="config", config_name="deterministic")
def main(cfg: DictConfig) -> None:
        
    # reused variables
    save_path = cfg['experiment']['save_path']
    
    # intial setup
    seed = cfg.get("training.seed", 42)  # Default to 42 if not specified
    utility.set_random_seeds(seed)
    
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
    
    # get model
    pred_model = model.get_unet_based_model(
        domain_x, domain_y, in_c, out_c, cfg
    )
    
    # force load from checkpoint
    assert cfg['experiment']['from_checkpoint'] is not None, 'Checkpoint is required!'
    model_utility.load_model_weights(pred_model, cfg['experiment']['from_checkpoint'])
    
    # get device
    device = utility.set_device(cfg['device']['preference'], cfg['device']['index'])
    
    # get feature lookup + switch
    feature_dict = model_utility.get_dict_of_features(cfg)
    feature_switch = model_utility.get_features_swtich(cfg)
    
    # get ground truth and predictions
    ground, predictions = evaluation.operator_prediction(
        train_dataset, 
        valid_dataset, 
        pred_model, 
        device, 
        feature_dict, 
        feature_switch,
        cfg['evaluation']['start_time'],
        cfg['evaluation']['iterations'],
        cfg
    )
    
    # Ensure the save directories exist
    trajectory_dir = os.path.join(cfg['evaluation']['save_path'], 'trajectories')
    rmse_dir = os.path.join(cfg['evaluation']['save_path'], 'rmse')

    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(rmse_dir, exist_ok=True)
    
    # save trajectory gifs
    evaluation.save_trajectory_gifs(
        ground,
        predictions,
        feature_dict,
        save_dir = trajectory_dir,
        cfg = cfg
    )
        
    # save rmse plots
    evaluation.save_rmse_plots(
        ground,
        predictions,
        feature_dict,
        save_dir = rmse_dir,
        cfg = cfg
    )


if __name__ == "__main__":
    main()