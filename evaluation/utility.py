import torch
from typing import Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from omegaconf import DictConfig

from torch.utils.data import Dataset
from torch.nn import Module


@torch.no_grad()
def operator_prediction(
    train_dataset: Dataset, 
    valid_dataset: Dataset, 
    pred_model: Module, 
    device: torch.device, 
    feature_dict: dict, 
    feature_switch: dict,
    start_timestep: int,
    n_timesteps: int,
    cfg: DictConfig = {}
) -> Tuple[np.ndarray, np.ndarray]:
    
    assert (start_timestep + n_timesteps) < len(valid_dataset), 'The predictions must be within the valid dataset!'
    
    # get only the timesteps indicated in config
    enabled_indices = [
        index 
        for feature, index in feature_dict.items() 
        if feature_switch.get(feature, False)  # Use `get` to avoid KeyError, default to False
    ]
        
    # get the mu and sigma for each prediction
    mu_sigma = [
        [valid_dataset[start_timestep+i][-1][0][enabled_indices], valid_dataset[start_timestep+i][-1][1][enabled_indices]]
        for i in range(n_timesteps)
    ]
    mu_sigma = np.array(mu_sigma)
        
    # get ground truth configuration
    ground = [
        valid_dataset[start_timestep+i][0][:85][enabled_indices] * mu_sigma[i][1] + mu_sigma[i][0]  # scale
        for i in range(n_timesteps)
    ]
    ground = np.array(ground)
    
    # move model to device
    pred_model = pred_model.to(device)
    
    # predict
    x, y, _, _ = valid_dataset[start_timestep]
    predictions = np.zeros((
        n_timesteps, len(enabled_indices), y.shape[1], y.shape[2])
    )
    
    predictions[0] = x[:85][enabled_indices] * mu_sigma[0][1] + mu_sigma[0][0]
    x = x.unsqueeze(0).to(device)

    # generate the forecast
    for i in range(n_timesteps-1):
        y = pred_model(x, torch.ones(1).to(device))[0]
        y = train_dataset._standardize(y)
        predictions[i+1] = y[0][enabled_indices].cpu() * mu_sigma[i+1][1] + mu_sigma[i+1][0]  # scale

        x_f, _, _, _ = valid_dataset[start_timestep+i+1]  # timestep to steal forcings from (im pretty sure they are exogenous)
        x_f = x_f.unsqueeze(0).to(device)

        x = torch.cat([y, x_f[:, 85:]], dim=1)  # concat forcings + constants for next timestep
        
    return ground, predictions


def seq_animation(sequence, title, save_root=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.axis('off')

    # Initialize with the first frame
    x_0_ind = sequence[0]
    im = ax.imshow(x_0_ind, origin='lower')
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # To store contour collections
    current_contours = []

    def update_frame(i):
        x_ind = sequence[i]
        
        # Update imshow data
        im.set_data(x_ind)
        im.set_clim(vmin=x_ind.min(), vmax=x_ind.max())  # Adjust color limits to current frame
        
        # Remove previous contours
        for coll in current_contours:
            coll.remove()
        current_contours.clear()
        
        # Create new filled contours (customize levels/colors as needed)
        contour_set = ax.contourf(x_ind, cmap='viridis', levels=20, alpha=0.75)  # Adjust levels and colormap as needed
        current_contours.extend(contour_set.collections)
        
        return [im] + current_contours

    ani = animation.FuncAnimation(fig, update_frame, frames=len(sequence), interval=1, blit=True)
    
    print(f'Generating animation.')
    output_path = f'{title}.gif' if save_root is None else os.path.join(save_root, f'{title}.gif')
    ani.save(output_path, writer='Pillow', fps=3)


def save_trajectory_gifs(
    ground: np.array,
    pred: np.array,
    feature_dict: dict,
    save_dir: str,
    cfg: DictConfig = {}
) -> None:

    for key, value in feature_dict.items():
        seq_animation(ground[:, value], f'{key}_ground', save_dir)
        plt.clf()
        plt.close()
        seq_animation(pred[:, value], f'{key}_pred', save_dir)
        plt.clf()
        plt.close()
        

def save_rmse_plots(
    ground: np.array,
    pred: np.array,
    feature_dict: dict,
    save_dir: str,
    cfg: DictConfig = {}
) -> None:

    for key, value in feature_dict.items():
        diff = ground[:, value] - pred[:, value] # calculate rmse
        mse = np.mean(diff**2, axis=(1, 2))  
        rmse = np.sqrt(mse)
        
        # Plot RMSE with dots and lines
        plt.plot(rmse, 'o-', label='RMSE')

        # Add title and axis labels
        plt.title(key)
        plt.xlabel('Predicted Timestep')
        plt.ylabel('RMSE')

        # Save the plot
        output_path = f'{key}.png' if save_dir is None else os.path.join(save_dir, f'{key}.png')
        plt.savefig(output_path)

        # Clear the plot for the next iteration
        plt.clf()
        plt.close()
