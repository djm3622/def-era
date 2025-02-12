import torch
from typing import Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from torch.utils.data import Dataset
from torch.nn import Module

from diffusers import DPMSolverMultistepScheduler


@torch.no_grad()
def prediction(
    train: Dataset, 
    valid: Dataset, 
    det_model: Module, 
    dif_model: Module,
    stra_params: dict, 
    sample_strategy: DPMSolverMultistepScheduler,
    device: torch.device, 
    walks: int,
    feature_dict: dict, 
    feature_switch: dict,
    start_time: float,
    iterations: int,
    cfg: DictConfig = {}
) -> Tuple[torch.Tensor]:
    
    assert (start_time + iterations) < len(valid), 'The predictions must be within the valid dataset!'
    samples = cfg['evaluation']['samples']
    
    # get only the timesteps indicated in config
    enabled_indices = [
        index 
        for feature, index in feature_dict.items() 
        if feature_switch.get(feature, False)  # Use `get` to avoid KeyError, default to False
    ]
        
    # get the mu and sigma for each prediction
    mu_sigma = [
        [valid[start_time+i][-1][0][enabled_indices].unsqueeze(0).repeat(samples, 1, 1, 1), 
         valid[start_time+i][-1][1][enabled_indices].unsqueeze(0).repeat(samples, 1, 1, 1)]
        for i in range(iterations)
    ]
    mu_sigma = np.array(mu_sigma)
    
    # move model to device
    det_model = det_model.to(device)
    dif_model = dif_model.to(device)
    
    # predict
    x, y, _, _ = valid[start_time]
    predictions = np.zeros((
        iterations, samples, len(enabled_indices), y.shape[1], y.shape[2])
    )
    
    x = x.unsqueeze(0).repeat(samples, 1, 1, 1)
    predictions[0] = x[:, :85][:, enabled_indices] * mu_sigma[0][1] + mu_sigma[0][0]
    x = x.to(device)

    # generate the forecast
    for i in range(iterations-1):
        
        # perturb with diffusion
        for _ in range(walks):
            stra_params['condition'] = x[:, :85]
            x[:, :85] = sample_strategy(**stra_params)[0]
        
        # det prediction
        y = det_model(x, torch.ones(1).to(device))[0]
        y = train._standardize(y)
        predictions[i+1] = y[:, enabled_indices].cpu() * mu_sigma[i+1][1] + mu_sigma[i+1][0]  # scale

        x_f, _, _, _ = valid[start_time+i+1]  # timestep to steal forcings from (im pretty sure they are exogenous)
        x_f = x_f.unsqueeze(0).repeat(samples, 1, 1, 1).to(device)

        x = torch.cat([y, x_f[:, 85:]], dim=1)  # concat forcings + constants for next timestep
        
    return predictions


def plot_metric(
    metrics: Tuple, 
    feature_dict: dict,
    name: str,
    var_names: Tuple,
    save_dir: str
) -> None:
    
    assert len(metrics) == len(var_names), f"{len(metrics)}, {len(var_names)}"
    
    plt.figure(figsize=(10, 6))
    
    for feature_name, feature_idx in feature_dict.items():
        
        for var_name, metric in zip(var_names, metrics):
            
            plt.plot(metric[:, feature_idx], 'o-', label=var_name)

        # Add title and axis labels
        plt.xlabel('Timestep')
        plt.legend()
        plt.title(f'{feature_name} {name}')
        plt.grid()

        # Save the plot
        plt.savefig(os.path.join(save_dir, f'{feature_name}_{name}.png'))

        # Clear the plot for the next iteration
        plt.clf()
        plt.close()


def compute_metrics(
    ground: np.array,
    det_predictions: np.array,
    diff_predictions: np.array,
    feature_dict: dict,
    save_dir: str,
    prob_metrics: list,
    det_metrics: list,
    cfg: DictConfig = {}
) -> None:
    
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(save_dir, exist_ok=True)
    
    ground = torch.from_numpy(ground)
    det_predictions = torch.from_numpy(det_predictions)
    diff_predictions = torch.from_numpy(diff_predictions)
        
    # compute probabilstic metrics
    for metric_name, metric in prob_metrics.items():

        if metric_name in ['energy', 'crps']:
            plot_metric(
                metric.compute(diff_predictions, ground),
                feature_dict,
                metric_name,
                ['spread', 'skill', 'score'],
                save_dir
            )

        else:
            plot_metric(
                [metric.compute(diff_predictions, ground), det_metrics['rmse'](det_predictions, ground)],
                feature_dict,
                metric_name,
                ['diffusion', 'deterministic'],
                save_dir
            )

    
def save_history(
    ground: np.array,
    det_predictions: np.array,
    diff_predictions: np.array,
    save_dir: str
) -> None:
    
    names = ['truth', 'deterministic', 'diffusion']
    
    # Create root directory if it doesn't exist
    root = os.path.join(save_dir, 'history')
    os.makedirs(root, exist_ok=True)
    
    # Save each array to a separate file
    for name, arr in zip(names, [ground, det_predictions, diff_predictions]):
        # Convert numpy array to tensor if needed
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
            
        # Create save path
        save_path = os.path.join(root, f'{name}.pth')
        
        # Save tensor
        torch.save(arr, save_path)