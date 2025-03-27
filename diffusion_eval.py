import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional

import torch
import random
import os
import numpy as np
from safetensors.torch import load_file

import data.era5_dataset as det_data
import data.era5_diffusion_dataset as diff_data

import evaluation.utility as evaluation
import evaluation.diff_utility as diff_eval

import model.deterministic.model as model_det
import model.diffusion.model as model_dif
import model.utility as model_utility
import model.diffusion.sampler_utility as sampler_utility

from model.objectives.energyscore import EnergyScore
from model.objectives.crps import CRPS
from model.objectives.meanrmse import EnsembleMeanRMSE

import utils.utility as utility

@hydra.main(version_base=None, config_path="config", config_name="diffusion")
def main(cfg: DictConfig) -> None:
        
    # reused variables
    save_path = cfg['evaluation']['save_path']
    os.makedirs(save_path, exist_ok=True)
    
    # get deterministic dataset
    train_dataset = det_data.ERA5Dataset(
        root_dir=cfg['dataset']['root_dir'],
        start_date=cfg['training']['dataset']['start_date'],
        end_date=cfg['training']['dataset']['end_date'],
        forecast_steps=cfg['dataset']['forecast_steps'],
        cfg=cfg,
    )
    valid_dataset = det_data.ERA5Dataset(
        root_dir=cfg['dataset']['root_dir'],
        start_date=cfg['training']['validation_dataset']['start_date'],
        end_date=cfg['training']['validation_dataset']['end_date'],
        forecast_steps=cfg['dataset']['forecast_steps'],
        cfg=cfg,
    )
    
    sample_x, sample_y, _, _ = train_dataset[0]
    
    in_c, domain_x, domain_y = sample_x.shape
    out_c, _, _ = sample_y.shape
    
    # get deterministic model
    det_model = model_det.get_unet_based_model(
        domain_x, domain_y, in_c, out_c, cfg
    )
    
    # force deterministic load from checkpoint
    assert cfg['deterministic']['from_state'] is not None, 'Checkpoint is required!'
    model_utility.load_model_weights(det_model, cfg['deterministic']['from_state'])
    print('Deterministic model loaded!')
    
    # get diffusion model
    diffusion_model = model_dif.get_unet_based_model(
        x = domain_x, y = domain_y,
        channels = out_c, cfg = cfg
    )
    
    # force diffusion model load from checkpoint
    assert cfg['diffusion']['from_state'] is not None, 'Checkpoint is required!'
    model_utility.load_model_weights(diffusion_model, cfg['diffusion']['from_state'])
    print('Diffusion model loaded!')
    
    # assert samplers check
    assert cfg['evaluation']['sampler'] in ['DDIM', 'DDPM', 'DPM++'], 'Sampler must be \'DDIM\', \'DDPM\', or \'DPM++\'!'
    assert cfg['evaluation']['num_steps'] > 0 and cfg['evaluation']['num_steps'] < cfg['dataset']['timestep'], 'Fix reduced steps!'
    
    # get device
    device = utility.set_device(cfg['device']['preference'], cfg['device']['index'])
    
    # get beta/alpha/alpha_bar
    beta, alpha, alpha_bar = utility.get_alpha_beta_bar(
        beta_start = cfg['dataset']['beta_start'],
        beta_end = cfg['dataset']['beta_end'],
        timesteps = cfg['dataset']['timestep'],
        device = device
    )
    
    # get sampler and config params (these are full params besides the condition param)
    sampler_params, sampler = sampler_utility.get_sampler(
        cfg['evaluation']['sampler'], 
        model = diffusion_model, 
        beta = beta,
        alpha = alpha,
        alpha_bar = alpha_bar,
        device = device,
        cfg = cfg
    )
    
    # get feature lookup + switch
    feature_dict = model_utility.get_dict_of_features(cfg)
    feature_switch = model_utility.get_features_swtich(cfg)
    
    # get ground truth and deterministic predictions
    ground, predictions = evaluation.operator_prediction(
        train_dataset, 
        valid_dataset, 
        det_model, 
        device, 
        feature_dict, 
        feature_switch,
        cfg['evaluation']['start_time'],
        cfg['evaluation']['iterations'],
        cfg
    )
    print('Deterministic predictions & ground truth generated!')
    
    # get diffusion predictions
    diff_predictions = diff_eval.prediction(
        train = train_dataset, 
        valid = valid_dataset, 
        det_model = det_model, 
        dif_model = diffusion_model,
        stra_params = sampler_params, 
        sample_strategy = sampler,
        device = device, 
        walks = cfg['evaluation']['walks'],
        feature_dict = feature_dict, 
        feature_switch = feature_switch,
        start_time = cfg['evaluation']['start_time'],
        iterations = cfg['evaluation']['iterations'],
        cfg = cfg
    )
    
    # metrics
    probalistic_metrics = {
        'energy': EnergyScore(),
        'crps': CRPS(),
        'mean-rmse': EnsembleMeanRMSE()
    }
    deterministic_metrics = {
        'rmse': lambda x, y: torch.sqrt(torch.mean((x - y)**2, dim=(-2, -1))) 
    }
        
    # compute and save metrics
    diff_eval.compute_metrics(
        ground = ground,
        det_predictions = predictions,
        diff_predictions = diff_predictions,
        feature_dict = feature_dict,
        save_dir = save_path,
        prob_metrics = probalistic_metrics,
        det_metrics = deterministic_metrics,
        cfg = cfg
    )
    
    # plot actual predictionss
    diff_eval.plot_ensemble_predictions(
        diff_predictions, 
        ground,
        feature_dict, 
        feature_switch, 
        save_dir = save_path, 
        forecast_hours = 6
    )

    # create animations
    diff_eval.create_ensemble_animation_pub(
        diff_predictions,
        feature_dict,
        feature_switch,
        save_dir=save_path,
        forecast_hours=6
    )
    
    # comparison animations
    diff_eval.create_comparison_animation(
        diff_predictions,
        ground,
        feature_dict,
        feature_switch,
        save_dir=save_path,
        forecast_hours=6
    )
    
    # member distances
    diff_eval.plot_ensemble_rmse(
        diff_predictions, 
        ground,
        predictions,
        feature_dict,
        feature_switch,
        save_dir=save_path,
        forecast_hours=6,
        plot_mean_rmse=True
    )
    
    # anaimation static
    diff_eval.plot_ensemble_snapshots(
        diff_predictions,
        feature_dict,
        feature_switch,
        save_dir = save_path,
        lead_times=[6, 40, 80, 120, 160, 200]
    )
    
    # comparision animation static
    diff_eval.plot_ensemble_comparison_snapshots(
        diff_predictions,
        ground,
        feature_dict,
        feature_switch,
        save_dir=save_path,
        forecast_hours=6
    )

    # Plot all ensemble members at specific lead times
    diff_eval.plot_all_ensemble_members(
        diff_predictions,
        feature_dict,
        feature_switch,
        save_dir = save_path,
        lead_times = [0, 40, 80, 120, 160, 200]
        
    )
    print('Plots generated!')
    
    # save the full history
    diff_eval.save_history(
        ground, predictions, diff_predictions, save_path
    )
    print('History saved!')
    

if __name__ == "__main__":
    main()