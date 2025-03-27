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


def seq_animation_pub(
    sequence: np.ndarray, 
    title: str, 
    save_root: str = None,
    cmap: str = 'viridis',
    fps: int = 3,
    contour_levels: int = 20,
    dpi: int = 100
) -> None:
    """
    Generate a publication-quality animation of a sequence with contour overlay.
    
    Args:
        sequence: 3D array of shape (time, height, width)
        title: Title for the animation
        save_root: Directory to save animation
        cmap: Colormap to use
        fps: Frames per second
        contour_levels: Number of contour levels
        dpi: Resolution for saved animation
    """
    # Set publication-ready styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Initialize with the first frame
    x_0_ind = sequence[0]
    
    # Calculate global min/max for consistent colormap
    vmin, vmax = np.min(sequence), np.max(sequence)
    
    # Create initial image
    im = ax.imshow(x_0_ind, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(title.split('_')[0].capitalize(), fontweight='bold')
    
    # Calculate contour levels for consistent contours
    levels = np.linspace(vmin, vmax, contour_levels)
    
    # Add title with frame counter
    title_obj = ax.set_title(f"{title.replace('_', ' ').title()} - Lead Time 0h", fontweight='bold')
    
    # Store contour collections
    current_contours = []
    
    def update_frame(i):
        x_ind = sequence[i]
        
        # Update image data (but keep color limits consistent)
        im.set_data(x_ind)
        
        # Update title with frame counter
        title_obj.set_text(f"{title.replace('_', ' ').title()} - Lead Time {i*6}h")
        
        # Remove previous contours
        for coll in current_contours:
            coll.remove()
        current_contours.clear()
        
        # Create new contours with consistent levels
        contour_set = ax.contour(x_ind, levels=levels, colors='black', alpha=0.5, linewidths=0.8)
        # Add filled contours with transparency
        contourf_set = ax.contourf(x_ind, levels=levels, cmap=cmap, alpha=0.7)
        
        current_contours.extend(contour_set.collections)
        current_contours.extend(contourf_set.collections)
        
        return [im, title_obj] + current_contours
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(sequence), interval=200, blit=True)
    
    print(f'Generating animation: {title}')
    output_path = f'{title}.gif' if save_root is None else os.path.join(save_root, f'{title}.gif')
    ani.save(output_path, writer='Pillow', fps=fps, dpi=dpi)
    plt.close()


def save_trajectory_gifs_pub(
    ground: np.ndarray,
    pred: np.ndarray,
    feature_dict: dict,
    save_dir: str,
    fps: int = 3,
    dpi: int = 150
) -> None:
    """
    Generate publication-quality animations comparing ground truth and predictions.
    
    Args:
        ground: Ground truth data of shape (time, features, height, width)
        pred: Prediction data of shape (time, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        save_dir: Directory to save animations
        fps: Frames per second
        dpi: Resolution for saved animations
    """
    # Ensure the save directories exist
    anim_dir = os.path.join(save_dir, 'animations')
    os.makedirs(anim_dir, exist_ok=True)
    
    # Select appropriate colormaps for different variables
    weather_cmaps = {
        'temperature': 'RdBu_r',
        'temp': 'RdBu_r',
        'pressure': 'viridis',
        'geopotential': 'magma',
        'wind': 'cool',
        'humidity': 'BrBG',
        'precipitation': 'Blues',
        'cloud': 'Greys',
    }
    
    for key, value in feature_dict.items():
        # Select colormap based on variable name
        cmap = next((weather_cmaps[k] for k in weather_cmaps if k in key.lower()), 'viridis')
        
        # Create animations
        seq_animation_pub(
            ground[:, value], 
            f'{key}_ground', 
            anim_dir,
            cmap=cmap,
            fps=fps,
            dpi=dpi
        )
        
        seq_animation_pub(
            pred[:, value], 
            f'{key}_prediction', 
            anim_dir,
            cmap=cmap,
            fps=fps,
            dpi=dpi
        )
        
        # Create error animation
        error = np.abs(ground[:, value] - pred[:, value])
        seq_animation_pub(
            error, 
            f'{key}_error', 
            anim_dir,
            cmap='Reds',
            fps=fps,
            dpi=dpi
        )


def save_rmse_plots_pub(
    ground: np.ndarray,
    pred: np.ndarray,
    feature_dict: dict,
    save_dir: str,
    forecast_hours: int = 6
) -> None:
    """
    Generate publication-quality RMSE plots for deterministic predictions.
    
    Args:
        ground: Ground truth data of shape (time, features, height, width)
        pred: Prediction data of shape (time, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        save_dir: Directory to save plots
        forecast_hours: Hours between each forecast step
    """
    # Ensure the save directories exist
    rmse_dir = os.path.join(save_dir, 'rmse')
    os.makedirs(rmse_dir, exist_ok=True)
    
    # Set publication-ready styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create time array
    time_steps = np.arange(len(ground)) * forecast_hours
    
    # Create a figure with all features for comparison
    plt.figure(figsize=(12, 8))
    
    for key, value in feature_dict.items():
        # Calculate RMSE
        diff = ground[:, value] - pred[:, value]
        mse = np.mean(diff**2, axis=(1, 2))  
        rmse = np.sqrt(mse)
        
        # Create display name
        display_name = key.replace('_', ' ').title()
        
        # Plot RMSE with scatter points and lines
        plt.plot(time_steps, rmse, 'o-', linewidth=2, label=display_name)
    
    # Add title and axis labels
    plt.title("Forecast Error by Variable", fontweight='bold')
    plt.xlabel('Forecast Lead Time (hours)', fontweight='bold')
    plt.ylabel('RMSE', fontweight='bold')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(os.path.join(rmse_dir, "combined_rmse.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(rmse_dir, "combined_rmse.pdf"), bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each feature
    for key, value in feature_dict.items():
        plt.figure(figsize=(10, 6))
        
        # Calculate RMSE
        diff = ground[:, value] - pred[:, value]
        mse = np.mean(diff**2, axis=(1, 2))  
        rmse = np.sqrt(mse)
        
        # Create display name
        display_name = key.replace('_', ' ').title()
        
        # Plot RMSE with scatter points and lines
        plt.plot(time_steps, rmse, 'o-', color='#1f77b4', linewidth=2, markersize=8)
        
        # Add title and axis labels
        plt.title(f"{display_name} Forecast Error", fontweight='bold')
        plt.xlabel('Forecast Lead Time (hours)', fontweight='bold')
        plt.ylabel('RMSE', fontweight='bold')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig(os.path.join(rmse_dir, f"{key}_rmse.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(rmse_dir, f"{key}_rmse.pdf"), bbox_inches='tight')
        plt.close()


def plot_error_snapshots(
    ground: np.ndarray,
    pred: np.ndarray,
    feature_dict: dict,
    save_dir: str,
    timesteps: list = None,
    num_panels: int = 4,
    forecast_hours: int = 6
) -> None:
    """
    Creates snapshot panels showing prediction and error at specific timesteps.
    
    Args:
        ground: Ground truth data of shape (time, features, height, width)
        pred: Prediction data of shape (time, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        save_dir: Directory to save plots
        timesteps: Specific timesteps to plot. If None, will select evenly spaced timesteps
        num_panels: Number of panels to include if timesteps is None
        forecast_hours: Hours between each forecast step
    """
    # Ensure the save directories exist
    snapshot_dir = os.path.join(save_dir, 'snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Get total number of timesteps
    iterations = len(ground)
    
    # Set publication-ready styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Select timesteps to display
    if timesteps is None:
        if iterations <= num_panels:
            timesteps = range(iterations)
        else:
            # Select evenly spaced timesteps
            timesteps = np.linspace(0, iterations-1, num_panels, dtype=int)
    
    # Select appropriate colormaps for different variables
    weather_cmaps = {
        'temperature': 'RdBu_r',
        'temp': 'RdBu_r',
        'pressure': 'viridis',
        'geopotential': 'magma',
        'wind': 'cool',
        'humidity': 'BrBG',
        'precipitation': 'Blues',
        'cloud': 'Greys',
    }
    
    # For each feature
    for key, value in feature_dict.items():
        # Create a more readable feature name
        display_name = key.replace('_', ' ').title()
        
        # Select colormap based on variable name
        cmap = next((weather_cmaps[k] for k in weather_cmaps if k in key.lower()), 'viridis')
        
        # Create figure with 3 rows (truth, prediction, error) and timesteps columns
        fig, axes = plt.subplots(3, len(timesteps), figsize=(4*len(timesteps), 12))
        
        # Calculate global min/max for consistent colormaps
        vmin, vmax = min(np.min(ground[:, value]), np.min(pred[:, value])), max(np.max(ground[:, value]), np.max(pred[:, value]))
        
        # Calculate error for all timesteps
        error = np.abs(ground[:, value] - pred[:, value])
        error_max = np.max(error)
        
        # Plot each timestep
        for i, t in enumerate(timesteps):
            # Plot ground truth (top row)
            im_truth = axes[0, i].imshow(
                ground[t, value], 
                origin='lower', 
                cmap=cmap,
                vmin=vmin, 
                vmax=vmax
            )
            axes[0, i].set_title(f'T+{t*forecast_hours}h')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # Add contour lines to truth
            contour_truth = axes[0, i].contour(
                ground[t, value],
                colors='black',
                alpha=0.5,
                levels=8,
                linewidths=0.5
            )
            
            # Plot prediction (middle row)
            im_pred = axes[1, i].imshow(
                pred[t, value], 
                origin='lower', 
                cmap=cmap,
                vmin=vmin, 
                vmax=vmax
            )
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            
            # Add contour lines to prediction
            contour_pred = axes[1, i].contour(
                pred[t, value],
                colors='black',
                alpha=0.5,
                levels=8,
                linewidths=0.5
            )
            
            # Plot absolute error (bottom row)
            im_error = axes[2, i].imshow(
                error[t], 
                origin='lower', 
                cmap='Reds',
                vmin=0, 
                vmax=error_max
            )
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
        
        # Add row labels
        axes[0, 0].set_ylabel('Ground Truth', fontweight='bold')
        axes[1, 0].set_ylabel('Prediction', fontweight='bold')
        axes[2, 0].set_ylabel('Absolute Error', fontweight='bold')
        
        # Add colorbars
        cbar_ax_field = fig.add_axes([0.92, 0.4, 0.02, 0.5])
        cbar_ax_error = fig.add_axes([0.92, 0.1, 0.02, 0.2])
        
        fig.colorbar(im_truth, cax=cbar_ax_field, label=display_name)
        fig.colorbar(im_error, cax=cbar_ax_error, label='Absolute Error')
        
        # Add overall title
        fig.suptitle(f'{display_name} Forecast Evaluation', fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save figure
        fig.savefig(os.path.join(snapshot_dir, f'{key}_snapshots.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(snapshot_dir, f'{key}_snapshots.pdf'), bbox_inches='tight')
        
        plt.close(fig)