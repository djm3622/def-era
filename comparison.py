import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Dict, Any
import torch
import random
import os
import numpy as np
from safetensors.torch import load_file
import evaluation.compare_utility as compare
import utils.utility as utility
import model.utility as model_utility


@hydra.main(version_base=None, config_path="config", config_name="comparison")
def main(cfg: DictConfig) -> None:
    
    # assert the names == history
    assert len(cfg['runs']['names']) == len(cfg['runs']['ckpts']), 'Number of names must equal number of checkpoints!'
    
    # reused variables
    names = cfg['runs']['names']
    checkpoint_dirs = cfg['runs']['ckpts']
    save_path = cfg['evaluation']['save_path']
    
    # set up save directory
    os.makedirs(save_path, exist_ok=True)
    
    # read in all the directories, associate them with names
    run_data = {}
    for name, ckpt_dir in zip(names, checkpoint_dirs):
        print(f"Loading data for {name} from {ckpt_dir}")
        
        # ensure the checkpoint directory exists
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist!")
            
        # from each directory get the required files
        det_path = os.path.join(ckpt_dir, "deterministic.pth")
        diff_path = os.path.join(ckpt_dir, "diffusion.pth")
        truth_path = os.path.join(ckpt_dir, "truth.pth")
        
        # verify all required files exist
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Deterministic model file not found at {det_path}")
        if not os.path.exists(diff_path):
            raise FileNotFoundError(f"Diffusion model file not found at {diff_path}")
        if not os.path.exists(truth_path):
            raise FileNotFoundError(f"Ground truth file not found at {truth_path}")
            
        # read in the data from each file
        run_data[name] = {
            'deterministic': torch.load(det_path, map_location='cpu', weights_only=False),
            'diffusion': torch.load(diff_path, map_location='cpu', weights_only=False),
            'truth': torch.load(truth_path, map_location='cpu', weights_only=False)
        }
        
        print(f"Successfully loaded data for {name}")
    
    # get feature lookup + switch from config
    feature_dict = model_utility.get_dict_of_features(cfg)
    feature_switch = model_utility.get_features_swtich(cfg)
    
    # Only need one deterministic result since they're all the same
    first_run = next(iter(run_data.values()))
    deterministic_data = first_run['deterministic']
    
    # Update all runs to use the same deterministic prediction
    for name in run_data:
        run_data[name]['deterministic'] = deterministic_data
    
    # call compare.run_comparison
    print("Running comparison analysis...")
    compare.run_comparison(
        run_data=run_data,
        feature_dict=feature_dict,
        feature_switch=feature_switch,
        save_dir=save_path,
        cfg=cfg
    )
    
    print(f"Comparison complete! Results saved to {save_path}")

if __name__ == "__main__":
    main()