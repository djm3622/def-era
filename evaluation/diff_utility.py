import torch
from typing import Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import matplotlib.animation as animation

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


def plot_ensemble_predictions(
    predictions: np.ndarray,
    ground: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    forecast_hours: int = 6,
    plot_obs: np.ndarray = None,
    plot_mean: bool = True,
    title_prefix: str = "Ensemble Forecast"
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'ensemble')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get iterations and samples dimensions
    iterations, samples, n_features, height, width = predictions.shape
    
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
    
    # Get enabled feature indices and names
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
    # Create time array
    forecast_times = np.arange(iterations) * forecast_hours
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Compute domain average for each ensemble member across iterations
        for sample_idx in range(samples):
            # Extract data for this sample across all iterations, averaging over spatial dimensions
            sample_data = np.mean(predictions[:, sample_idx, feature_pos], axis=(1, 2))
            
            # Plot with low alpha
            plt.plot(forecast_times, sample_data, '-', alpha=0.3, color='gray', linewidth=0.8)
        
        # Plot ensemble mean if requested
        if plot_mean:
            ensemble_mean = np.mean(predictions[:, :, feature_pos], axis=1)
            mean_data = np.mean(ensemble_mean, axis=(1, 2))
            plt.plot(forecast_times, mean_data, 'b-', linewidth=3, label='Ensemble Mean')
        
        # Plot ground truth trajectory
        ground_data = np.mean(ground[:, feature_pos], axis=(1, 2))
        plt.plot(forecast_times, ground_data, 'r-', linewidth=3, label='Ground Truth')
        
        # Plot observations if provided (in addition to ground truth)
        if plot_obs is not None:
            obs_data = np.mean(plot_obs[:, feature_pos], axis=(1, 2))
            plt.plot(forecast_times, obs_data, 'k-', linewidth=3, label='Observation')
        
        # Add title and labels
        plt.title(f"{title_prefix}: {display_feature_name}")
        plt.xlabel('Forecast Lead Time (hours)', fontweight='bold')
        plt.ylabel(f'{display_feature_name} (Domain Average)', fontweight='bold')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'ensemble_{feature_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_ensemble_predictions(
    predictions: np.ndarray,
    ground: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    forecast_hours: int = 6,
    plot_obs: np.ndarray = None,
    plot_mean: bool = True,
    title_prefix: str = "Ensemble Forecast"
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'ensemble')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get iterations and samples dimensions
    iterations, samples, n_features, height, width = predictions.shape
    
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
    
    # Get enabled feature indices and names
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
    # Create time array
    forecast_times = np.arange(iterations) * forecast_hours
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Compute domain average for each ensemble member across iterations
        for sample_idx in range(samples):
            # Extract data for this sample across all iterations, averaging over spatial dimensions
            sample_data = np.mean(predictions[:, sample_idx, feature_pos], axis=(1, 2))
            
            # Plot with low alpha
            plt.plot(forecast_times, sample_data, '-', alpha=0.3, color='gray', linewidth=0.8)
        
        # Plot ensemble mean if requested
        if plot_mean:
            ensemble_mean = np.mean(predictions[:, :, feature_pos], axis=1)
            mean_data = np.mean(ensemble_mean, axis=(1, 2))
            plt.plot(forecast_times, mean_data, 'b-', linewidth=3, label='Ensemble Mean')
        
        # Plot ground truth trajectory
        ground_data = np.mean(ground[:, feature_pos], axis=(1, 2))
        plt.plot(forecast_times, ground_data, 'r-', linewidth=3, label='Ground Truth')
        
        # Plot observations if provided (in addition to ground truth)
        if plot_obs is not None:
            obs_data = np.mean(plot_obs[:, feature_pos], axis=(1, 2))
            plt.plot(forecast_times, obs_data, 'k-', linewidth=3, label='Observation')
        
        # Add title and labels
        plt.title(f"{title_prefix}: {display_feature_name}")
        plt.xlabel('Forecast Lead Time (hours)', fontweight='bold')
        plt.ylabel(f'{display_feature_name} (Domain Average)', fontweight='bold')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'ensemble_{feature_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_ensemble_snapshots(
    predictions: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    lead_times: list = None,
    forecast_hours: int = 6
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'snapshots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get iterations and samples dimensions
    iterations, samples, n_features, height, width = predictions.shape
    
    # Set default lead times if not provided
    if lead_times is None:
        lead_times = [0, 24, 48, 72]
    
    # Convert lead times to iteration indices
    lead_indices = [min(int(lt / forecast_hours), iterations-1) for lt in lead_times]
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
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
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(2, len(lead_indices), figsize=(4*len(lead_indices), 8))
        
        # Define global min/max for consistent colormaps
        all_means = np.array([np.mean(predictions[idx, :, feature_pos], axis=0) for idx in lead_indices])
        all_spreads = np.array([np.std(predictions[idx, :, feature_pos], axis=0) for idx in lead_indices])
        
        vmin_mean, vmax_mean = np.min(all_means), np.max(all_means)
        vmin_spread, vmax_spread = np.min(all_spreads), np.max(all_spreads)
        
        # Plot for each lead time
        for i, (idx, lead_time) in enumerate(zip(lead_indices, lead_times)):
            # Compute ensemble mean and spread
            mean = np.mean(predictions[idx, :, feature_pos], axis=0)
            spread = np.std(predictions[idx, :, feature_pos], axis=0)
            
            # Plot mean
            im1 = axes[0, i].imshow(mean, origin='lower', cmap='viridis', vmin=vmin_mean, vmax=vmax_mean)
            axes[0, i].set_title(f'Mean at {lead_time}h')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # Plot spread
            im2 = axes[1, i].imshow(spread, origin='lower', cmap='plasma', vmin=vmin_spread, vmax=vmax_spread)
            axes[1, i].set_title(f'Spread at {lead_time}h')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # Add colorbars
        cax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
        cax2 = fig.add_axes([0.92, 0.1, 0.02, 0.35])
        fig.colorbar(im1, cax=cax1, label=display_feature_name)
        fig.colorbar(im2, cax=cax2, label=f'{display_feature_name} Spread')
        
        # Add overall title
        fig.suptitle(f'{display_feature_name} Ensemble Forecast Evolution', fontweight='bold')
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'{feature_name}_snapshots.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{feature_name}_snapshots.pdf'), bbox_inches='tight')
        plt.close()


def plot_all_ensemble_members(
    predictions: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    lead_times: list = None,
    forecast_hours: int = 6,
    max_members: int = 16  # Limit the number of members to display
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'members')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get iterations and samples dimensions
    iterations, samples, n_features, height, width = predictions.shape
    
    # Set default lead times if not provided
    if lead_times is None:
        lead_times = [0, 24, 48, 72]
    
    # Convert lead times to iteration indices
    lead_indices = [min(int(lt / forecast_hours), iterations-1) for lt in lead_times]
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
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
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # For each lead time
        for idx, lead_time in zip(lead_indices, lead_times):
            # Determine number of members to plot (limit to max_members)
            num_members = min(samples, max_members)
            
            # Determine grid size for subplots
            grid_size = int(np.ceil(np.sqrt(num_members)))
            
            # Create figure
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(3*grid_size, 3*grid_size))
            
            # Make axes a 2D array if it's not already (happens when grid_size=1)
            if grid_size == 1:
                axes = np.array([[axes]])
            
            # Get data for this lead time
            data = predictions[idx, :num_members, feature_pos]
            
            # Get global min/max for consistent colormap
            vmin, vmax = data.min(), data.max()
            
            # Plot each member
            for m in range(num_members):
                row, col = m // grid_size, m % grid_size
                im = axes[row, col].imshow(data[m], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[row, col].set_title(f'Member {m+1}')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            
            # Hide empty subplots
            for m in range(num_members, grid_size*grid_size):
                row, col = m // grid_size, m % grid_size
                axes[row, col].axis('off')
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, label=display_feature_name)
            
            # Add overall title
            fig.suptitle(f'{display_feature_name} Ensemble Members at {lead_time}h Lead Time', fontweight='bold')
            
            # Adjust layout
            plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)
            
            # Save figure
            plt.savefig(os.path.join(save_dir, f'{feature_name}_{lead_time}h_members.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{feature_name}_{lead_time}h_members.pdf'), bbox_inches='tight')
            plt.close()


def create_spaghetti_plot(
    predictions: np.ndarray,
    ground: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    forecast_hours: int = 6,
    contour_levels: list = None
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'spaghetti')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get iterations and samples dimensions
    iterations, samples, n_features, height, width = predictions.shape
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
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
    
    # Create array of all lead times
    all_lead_times = np.arange(iterations) * forecast_hours
    
    # Select time steps for plotting
    selected_steps = [6, min(int(iterations/3), iterations-1), min(int(iterations*2/3), iterations-1), iterations-1]
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Determine contour levels if not provided
        if contour_levels is None:
            # Get range of values for this feature
            all_data = np.concatenate([
                predictions[:, :, feature_pos].reshape(-1),
                ground[:, feature_pos].reshape(-1)
            ])
            min_val, max_val = np.min(all_data), np.max(all_data)
            # Create evenly spaced contour levels
            contour_levels = np.linspace(min_val, max_val, 10)
        
        # Create figure with 2x2 grid for different lead times
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot for selected time steps
        for i, step in enumerate(selected_steps):
            if step >= iterations:
                # Skip if the step is beyond available data
                axes[i].set_visible(False)
                continue
                
            # Plot ensemble members as thin contour lines
            for member in range(samples):
                axes[i].contour(
                    predictions[step, member, feature_pos],
                    levels=contour_levels,
                    colors='blue',
                    linewidths=0.5,
                    alpha=0.3
                )
            
            # Plot ground truth as thick contour lines
            axes[i].contour(
                ground[step, feature_pos],
                levels=contour_levels,
                colors='red',
                linewidths=2
            )
            
            # Add title for this subplot - use the calculated lead time
            lead_time = all_lead_times[step]
            axes[i].set_title(f'Lead Time: {lead_time}h')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Add overall title
        fig.suptitle(f'{display_feature_name} Ensemble Forecast - Spaghetti Plot', fontweight='bold')
        
        # Add custom legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', alpha=0.3, linewidth=0.5),
            Line2D([0], [0], color='red', linewidth=2)
        ]
        fig.legend(
            custom_lines, 
            ['Ensemble Members', 'Ground Truth'], 
            loc='lower center', 
            ncol=2, 
            bbox_to_anchor=(0.5, 0.02)
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'{feature_name}_spaghetti.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{feature_name}_spaghetti.pdf'), bbox_inches='tight')
        plt.close()

        
def create_ensemble_animation(
    predictions: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    fps: int = 3
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'animations')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Compute ensemble mean for all iterations
        ensemble_mean = np.mean(predictions[:, :, feature_pos], axis=1)
        
        # Create animation of ensemble mean
        create_field_animation(
            ensemble_mean,
            f"{display_feature_name}_ensemble_mean",
            save_dir,
            fps=fps,
            cmap='viridis'
        )
        
        # Compute ensemble spread for all iterations
        ensemble_spread = np.std(predictions[:, :, feature_pos], axis=1)
        
        # Create animation of ensemble spread
        create_field_animation(
            ensemble_spread,
            f"{display_feature_name}_ensemble_spread",
            save_dir,
            fps=fps,
            cmap='plasma'
        )


def create_field_animation(
    sequence: np.ndarray,
    title: str,
    save_dir: str,
    fps: int = 3,
    cmap: str = 'viridis'
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')
    
    # Initialize with the first frame
    im = ax.imshow(sequence[0], origin='lower', cmap=cmap)
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    # To store contour collections
    current_contours = []
    
    def update_frame(i):
        x_ind = sequence[i]
        
        # Update imshow data
        im.set_data(x_ind)
        im.set_clim(vmin=x_ind.min(), vmax=x_ind.max())
        
        # Remove previous contours
        for coll in current_contours:
            coll.remove()
        current_contours.clear()
        
        # Create new contours
        contour_set = ax.contour(x_ind, colors='black', alpha=0.5, levels=10)
        current_contours.extend(contour_set.collections)
        
        ax.set_title(f'Forecast Lead Time: {i*6} hours', fontweight='bold')
        
        return [im] + current_contours
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(sequence), interval=200, blit=True)
    
    output_path = os.path.join(save_dir, f'{title}.gif')
    ani.save(output_path, writer='Pillow', fps=fps)
    plt.close()


def plot_metric(
    metrics: Tuple, 
    feature_dict: dict,
    name: str,
    var_names: Tuple,
    save_dir: str
) -> None:
    
    assert len(metrics) == len(var_names), f"{len(metrics)}, {len(var_names)}"
    
    plt.figure(figsize=(10, 6))
    
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
    
    for feature_name, feature_idx in feature_dict.items():
        
        # Create a more readable feature name by replacing underscores with spaces and title casing
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Create a more readable metric name
        display_metric_name = name.upper() if name.lower() in ['crps', 'rmse', 'mae'] else name.replace('_', ' ').title()
        
        for var_name, metric in zip(var_names, metrics):
            y_vals = metric[1:, feature_idx]
            x_vals = np.arange(1, y_vals.shape[0]+1) * 6
            plt.plot(x_vals, y_vals, 'o-', label=var_name)
            
        # Add title and axis labels
        plt.xlabel('Forecast Lead Time', fontweight='bold')
        
        # Create a professional title
        plt.title(f'{display_metric_name} for {display_feature_name}', fontweight='bold')
        
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(linestyle='--', alpha=0.7)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save the plot with high DPI for publication quality
        plt.savefig(os.path.join(save_dir, f'{feature_name}_{name}.png'), dpi=300, bbox_inches='tight')
        
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