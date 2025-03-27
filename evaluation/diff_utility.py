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
    """
    Creates publication-quality plots of all ensemble members at specified lead times.
    
    Args:
        predictions: Array of shape (iterations, samples, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        feature_switch: Dictionary indicating which features to plot
        save_dir: Directory to save plots
        lead_times: List of lead times to plot (in hours). If None, uses [0, 24, 48, 72]
        forecast_hours: Hours between each forecast step
        max_members: Maximum number of members to display (to avoid overcrowded plots)
    """
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
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Select colormap based on variable name
        cmap = next((weather_cmaps[k] for k in weather_cmaps if k in feature_name.lower()), 'viridis')
        
        # For each lead time
        for idx, lead_time in zip(lead_indices, lead_times):
            if idx >= iterations:
                print(f"Skipping lead time {lead_time}h (beyond available iterations)")
                continue
                
            # Determine number of members to plot (limit to max_members)
            num_members = min(samples, max_members)
            
            # Determine grid size for subplots (aim for square layout, but can adjust based on needs)
            grid_size = int(np.ceil(np.sqrt(num_members)))
            
            # Adjust for better aspect ratio if needed
            if grid_size**2 - num_members > grid_size:
                # If we would have more than one row of empty plots, use rectangular layout
                grid_cols = grid_size
                grid_rows = int(np.ceil(num_members / grid_cols))
            else:
                grid_rows = grid_cols = grid_size
            
            # Create figure
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*grid_cols, 3*grid_rows))
            
            # Make axes a 2D array if it's not already
            if grid_rows == 1 and grid_cols == 1:
                axes = np.array([[axes]])
            elif grid_rows == 1 or grid_cols == 1:
                axes = np.array(axes).reshape(grid_rows, grid_cols)
            
            # Get data for this lead time
            data = predictions[idx, :num_members, feature_pos]
            
            # Get global min/max for consistent colormap across all members
            vmin, vmax = data.min(), data.max()
            
            # Calculate contour levels
            levels = np.linspace(vmin, vmax, 10)
            
            # Compute ensemble mean for reference
            ensemble_mean = np.mean(data, axis=0)
            
            # Plot each member
            for m in range(num_members):
                row, col = m // grid_cols, m % grid_cols
                
                # Plot member data
                im = axes[row, col].imshow(data[m], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                
                # Add contour lines
                contour = axes[row, col].contour(
                    data[m], 
                    levels=levels, 
                    colors='black', 
                    alpha=0.5, 
                    linewidths=0.8
                )
                
                # Add title with member number
                axes[row, col].set_title(f'Member {m+1}')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            
            # Hide empty subplots
            for m in range(num_members, grid_rows * grid_cols):
                row, col = m // grid_cols, m % grid_cols
                axes[row, col].axis('off')
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(display_feature_name, fontweight='bold')
            
            # Add overall title
            fig.suptitle(
                f'{display_feature_name} Ensemble Members at {lead_time}h Lead Time', 
                fontweight='bold',
                fontsize=16
            )
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 0.9, 0.98])
            
            # Save figure in high resolution
            plt.savefig(
                os.path.join(save_dir, f'{feature_name}_{lead_time}h_members.png'), 
                dpi=300, 
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(save_dir, f'{feature_name}_{lead_time}h_members.pdf'), 
                bbox_inches='tight'
            )
            plt.close()
            
        # Create a combined figure showing ensemble members at multiple lead times
        # Only do this if we have a reasonable number of members and lead times
        if num_members <= 9 and len(lead_times) <= 4:
            # Create a large figure
            fig = plt.figure(figsize=(4*len(lead_times), 4*min(3, num_members)))
            
            # Create a grid that's lead_times wide and shows up to 3 members per lead time
            members_to_show = min(3, num_members)
            gs = fig.add_gridspec(members_to_show, len(lead_times))
            
            # Show a different subset of members for each feature to provide coverage
            member_indices = np.linspace(0, num_members-1, members_to_show, dtype=int)
            
            # For each lead time
            for lt_idx, (idx, lead_time) in enumerate(zip(lead_indices, lead_times)):
                if idx >= iterations:
                    continue
                    
                # Get data for this lead time
                data = predictions[idx, :, feature_pos]
                
                # Get global min/max for consistent colormap across all members and lead times
                vmin, vmax = data.min(), data.max()
                levels = np.linspace(vmin, vmax, 10)
                
                # Plot selected members
                for m_idx, member in enumerate(member_indices):
                    ax = fig.add_subplot(gs[m_idx, lt_idx])
                    
                    # Plot member data
                    im = ax.imshow(data[member], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                    
                    # Add contour lines
                    contour = ax.contour(
                        data[member], 
                        levels=levels, 
                        colors='black', 
                        alpha=0.5, 
                        linewidths=0.8
                    )
                    
                    # Configure axis
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Add titles only for top row and left column
                    if m_idx == 0:
                        ax.set_title(f'T+{lead_time}h', fontweight='bold')
                    if lt_idx == 0:
                        ax.set_ylabel(f'Member {member+1}', fontweight='bold')
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(display_feature_name, fontweight='bold')
            
            # Add overall title
            fig.suptitle(
                f'{display_feature_name} Ensemble Member Evolution', 
                fontweight='bold',
                fontsize=16
            )
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            
            # Save figure
            plt.savefig(
                os.path.join(save_dir, f'{feature_name}_member_evolution.png'), 
                dpi=300, 
                bbox_inches='tight'
            )
            plt.savefig(
                os.path.join(save_dir, f'{feature_name}_member_evolution.pdf'), 
                bbox_inches='tight'
            )
            plt.close()


def plot_ensemble_rmse(
    predictions: np.ndarray,  # ensemble predictions
    ground: np.ndarray,       # ground truth
    deterministic: np.ndarray, # deterministic predictions
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    forecast_hours: int = 6,
    plot_mean_rmse: bool = True,
    title_prefix: str = "Ensemble Forecast RMSE"
) -> None:
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'ermse')
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
        (feature, idx) for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
    # Create time array - skip iteration 0
    forecast_times = np.arange(1, iterations) * forecast_hours
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Initialize array to store RMSE for each member - skip iteration 0
        member_rmse = np.zeros((iterations-1, samples))
        
        # Compute RMSE for each ensemble member across iterations - skip iteration 0
        for sample_idx in range(samples):
            for i in range(1, iterations):
                # Calculate RMSE between this ensemble member and ground truth
                # for this feature at this time step
                pred = predictions[i, sample_idx, feature_pos]
                true = ground[i, feature_pos]
                member_rmse[i-1, sample_idx] = np.sqrt(np.mean((pred - true)**2))
        
            # Plot with low alpha
            plt.plot(forecast_times, member_rmse[:, sample_idx], '-', alpha=0.3, color='gray', linewidth=0.8)
        
        # Plot average RMSE across all members
        mean_member_rmse = np.mean(member_rmse, axis=1)
        plt.plot(forecast_times, mean_member_rmse, 'g-', linewidth=3, label='Average Member RMSE')
        
        # Plot ensemble mean RMSE if requested
        if plot_mean_rmse:
            # Calculate RMSE of ensemble mean compared to ground truth
            mean_rmse = np.zeros(iterations-1)
            for i in range(1, iterations):
                mean_rmse[i-1] = np.sqrt(np.mean(
                    (np.mean(predictions[i, :, feature_pos], axis=0) - ground[i, feature_pos])**2
                ))
            
            plt.plot(forecast_times, mean_rmse, 'b-', linewidth=3, label='Ensemble Mean RMSE')
        
        # Calculate and plot RMSE for deterministic predictions
        det_rmse = np.zeros(iterations-1)
        for i in range(1, iterations):
            det_rmse[i-1] = np.sqrt(np.mean(
                (deterministic[i, feature_pos] - ground[i, feature_pos])**2
            ))
        
        plt.plot(forecast_times, det_rmse, 'r-', linewidth=3, label='Deterministic RMSE')
        
        # Add title and labels
        plt.title(f"{title_prefix}: {display_feature_name}")
        plt.xlabel('Forecast Lead Time (hours)', fontweight='bold')
        plt.ylabel(f'RMSE', fontweight='bold')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'rmse_{feature_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        
def create_ensemble_animation_pub(
    predictions: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    forecast_hours: int = 6,
    fps: int = 3,
    dpi: int = 150
) -> None:
    """
    Creates publication-quality animations of ensemble mean and spread evolution.
    
    Args:
        predictions: Array of shape (iterations, samples, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        feature_switch: Dictionary indicating which features to plot
        save_dir: Directory to save animations
        forecast_hours: Hours between each forecast step
        fps: Frames per second
        dpi: Resolution for saved animations
    """
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'animations')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
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
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Select colormap based on variable name
        cmap = next((weather_cmaps[k] for k in weather_cmaps if k in feature_name.lower()), 'viridis')
        
        # Compute ensemble mean for all iterations
        ensemble_mean = np.mean(predictions[:, :, feature_pos], axis=1)
        
        # Create animation of ensemble mean
        create_field_animation_pub(
            ensemble_mean,
            f"{display_feature_name}_Ensemble_Mean",
            save_dir,
            forecast_hours=forecast_hours,
            fps=fps,
            cmap=cmap,
            dpi=dpi
        )
        
        # Compute ensemble spread for all iterations
        ensemble_spread = np.std(predictions[:, :, feature_pos], axis=1)
        
        # Create animation of ensemble spread
        create_field_animation_pub(
            ensemble_spread,
            f"{display_feature_name}_Ensemble_Spread",
            save_dir,
            forecast_hours=forecast_hours,
            fps=fps,
            cmap='plasma',  # Always use plasma for spread
            dpi=dpi
        )


def create_field_animation_pub(
    sequence: np.ndarray,
    title: str,
    save_dir: str,
    forecast_hours: int = 6,
    fps: int = 3,
    cmap: str = 'viridis',
    dpi: int = 150
) -> None:
    """
    Creates a publication-quality animation of a field evolving over time.
    
    Args:
        sequence: Array of field values (iterations, height, width)
        title: Title for the animation
        save_dir: Directory to save animation
        forecast_hours: Hours between each forecast step
        fps: Frames per second
        cmap: Colormap to use
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
    
    # Calculate global min/max for consistent colormap
    vmin, vmax = np.min(sequence), np.max(sequence)
    
    # Initialize with the first frame
    im = ax.imshow(sequence[0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(title.split('_')[0].replace('Ensemble', '').strip(), fontweight='bold')
    
    # Calculate contour levels for consistent contours across frames
    levels = np.linspace(vmin, vmax, 15)
    
    # Add title with frame counter
    title_obj = ax.set_title(f"{title.replace('_', ' ')} - Lead Time: 0h", fontweight='bold')
    
    # To store contour collections
    current_contours = []
    
    def update_frame(i):
        x_ind = sequence[i]
        
        # Update imshow data (color limits remain consistent)
        im.set_data(x_ind)
        
        # Update title with lead time
        title_obj.set_text(f"{title.replace('_', ' ')} - Lead Time: {i*forecast_hours}h")
        
        # Remove previous contours
        for coll in current_contours:
            coll.remove()
        current_contours.clear()
        
        # Create new contours with consistent levels
        contour_set = ax.contour(x_ind, levels=levels, colors='black', alpha=0.5, linewidths=0.8)
        # Add filled contours with transparency
        contourf_set = ax.contourf(x_ind, levels=levels, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)
        
        current_contours.extend(contour_set.collections)
        current_contours.extend(contourf_set.collections)
        
        return [im, title_obj] + current_contours
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(sequence), interval=200, blit=True)
    
    output_path = os.path.join(save_dir, f'{title}.gif')
    print(f'Generating animation: {output_path}')
    ani.save(output_path, writer='Pillow', fps=fps, dpi=dpi)
    plt.close()


def create_comparison_animation(
    predictions: np.ndarray,
    ground: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    forecast_hours: int = 6,
    fps: int = 3,
    dpi: int = 150
) -> None:
    """
    Creates animations comparing ensemble mean with ground truth and showing error.
    
    Args:
        predictions: Array of shape (iterations, samples, features, height, width)
        ground: Ground truth data of shape (iterations, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        feature_switch: Dictionary indicating which features to plot
        save_dir: Directory to save animations
        forecast_hours: Hours between each forecast step
        fps: Frames per second
        dpi: Resolution for saved animations
    """
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'comparison_animations')
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
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
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Select colormap based on variable name
        cmap = next((weather_cmaps[k] for k in weather_cmaps if k in feature_name.lower()), 'viridis')
        
        # Compute ensemble mean
        ensemble_mean = np.mean(predictions[:, :, feature_pos], axis=1)
        
        # Create figure with 2x2 subplots (truth, prediction, error, spread)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Calculate global min/max for consistent colormap for the field
        field_data = np.concatenate([ensemble_mean.flatten(), ground[:, feature_pos].flatten()])
        vmin, vmax = np.min(field_data), np.max(field_data)
        levels = np.linspace(vmin, vmax, 15)
        
        # Calculate error
        error = np.abs(ensemble_mean - ground[:, feature_pos])
        error_max = np.max(error)
        error_levels = np.linspace(0, error_max, 15)
        
        # Calculate spread
        ensemble_spread = np.std(predictions[:, :, feature_pos], axis=1)
        spread_max = np.max(ensemble_spread)
        spread_levels = np.linspace(0, spread_max, 15)
        
        # Initialize plots
        im_truth = axes[0].imshow(ground[0, feature_pos], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        im_pred = axes[1].imshow(ensemble_mean[0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        im_error = axes[2].imshow(error[0], origin='lower', cmap='Reds', vmin=0, vmax=error_max)
        im_spread = axes[3].imshow(ensemble_spread[0], origin='lower', cmap='plasma', vmin=0, vmax=spread_max)
        
        # Add titles
        title_truth = axes[0].set_title('Ground Truth', fontweight='bold')
        title_pred = axes[1].set_title('Ensemble Mean', fontweight='bold')
        title_error = axes[2].set_title('Absolute Error', fontweight='bold')
        title_spread = axes[3].set_title('Ensemble Spread', fontweight='bold')
        
        # Add lead time as main title
        main_title = fig.suptitle(f"{display_feature_name} - Lead Time: 0h", fontweight='bold', fontsize=16)
        
        # Initialize contours
        contour_collections = []
        
        # Add colorbars
        cbar_truth = fig.colorbar(im_truth, ax=axes[0])
        cbar_pred = fig.colorbar(im_pred, ax=axes[1])
        cbar_error = fig.colorbar(im_error, ax=axes[2])
        cbar_spread = fig.colorbar(im_spread, ax=axes[3])
        
        cbar_truth.set_label(display_feature_name, fontweight='bold')
        cbar_pred.set_label(display_feature_name, fontweight='bold')
        cbar_error.set_label('Absolute Error', fontweight='bold')
        cbar_spread.set_label('Spread', fontweight='bold')
        
        # Turn off axis ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        def update_frame(i):
            # Update main title with lead time
            main_title.set_text(f"{display_feature_name} - Lead Time: {i*forecast_hours}h")
            
            # Update images
            im_truth.set_data(ground[i, feature_pos])
            im_pred.set_data(ensemble_mean[i])
            im_error.set_data(error[i])
            im_spread.set_data(ensemble_spread[i])
            
            # Remove previous contours
            for coll in contour_collections:
                if coll in axes[0].collections or coll in axes[1].collections or \
                   coll in axes[2].collections or coll in axes[3].collections:
                    coll.remove()
            contour_collections.clear()
            
            # Add new contours
            contours_truth = axes[0].contour(ground[i, feature_pos], 
                                            levels=levels, colors='black', alpha=0.5, linewidths=0.8)
            contours_pred = axes[1].contour(ensemble_mean[i], 
                                           levels=levels, colors='black', alpha=0.5, linewidths=0.8)
            contours_error = axes[2].contour(error[i], 
                                            levels=error_levels, colors='black', alpha=0.5, linewidths=0.8)
            contours_spread = axes[3].contour(ensemble_spread[i], 
                                             levels=spread_levels, colors='black', alpha=0.5, linewidths=0.8)
            
            # Collect all contour collections
            for contour_set in [contours_truth, contours_pred, contours_error, contours_spread]:
                contour_collections.extend(contour_set.collections)
            
            return [im_truth, im_pred, im_error, im_spread, main_title] + contour_collections
        
        ani = animation.FuncAnimation(fig, update_frame, frames=len(ensemble_mean), interval=200, blit=True)
        
        output_path = os.path.join(save_dir, f'{feature_name}_comparison.gif')
        print(f'Generating comparison animation: {output_path}')
        ani.save(output_path, writer='Pillow', fps=fps, dpi=dpi)
        plt.close()


def plot_ensemble_comparison_snapshots(
    predictions: np.ndarray,
    ground: np.ndarray,
    feature_dict: dict,
    feature_switch: dict,
    save_dir: str,
    timesteps: list = None,
    num_panels: int = 4,
    forecast_hours: int = 6
) -> None:
    """
    Creates snapshot panels comparing ground truth, ensemble mean, error, and spread
    at specific lead times in a 4-panel layout similar to the comparison animation.
    
    Args:
        predictions: Array of shape (iterations, samples, features, height, width)
        ground: Ground truth data of shape (iterations, features, height, width)
        feature_dict: Dictionary mapping feature names to indices
        feature_switch: Dictionary indicating which features to plot
        save_dir: Directory to save plots
        timesteps: Specific timesteps to plot. If None, will select evenly spaced timesteps
        num_panels: Number of panels to include if timesteps is None
        forecast_hours: Hours between each forecast step
    """
    # Ensure the save directories exist
    save_dir = os.path.join(save_dir, 'comparison_snapshots')
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
    
    # Select timesteps to display
    if timesteps is None:
        if iterations <= num_panels:
            timesteps = range(iterations)
        else:
            # Select evenly spaced timesteps
            timesteps = np.linspace(0, iterations-1, num_panels, dtype=int)
    
    # Get enabled feature indices
    enabled_features = [
        (feature, idx) 
        for feature, idx in feature_dict.items() 
        if feature_switch.get(feature, False)
    ]
    
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
    
    # For each enabled feature
    for feature_name, feature_idx in enabled_features:
        # Find the feature's position in the enabled indices list
        feature_pos = list(feature_dict.values()).index(feature_idx)
        
        # Create a more readable feature name
        display_feature_name = feature_name.replace('_', ' ').title()
        
        # Select colormap based on variable name
        cmap = next((weather_cmaps[k] for k in weather_cmaps if k in feature_name.lower()), 'viridis')
        
        # Compute ensemble mean and spread
        ensemble_mean = np.mean(predictions[:, :, feature_pos], axis=1)
        ensemble_spread = np.std(predictions[:, :, feature_pos], axis=1)
        
        # Calculate error
        abs_error = np.abs(ensemble_mean - ground[:, feature_pos])
        
        # Calculate global min/max for consistent colormaps
        field_data = np.concatenate([ensemble_mean.flatten(), ground[:, feature_pos].flatten()])
        vmin_field, vmax_field = np.min(field_data), np.max(field_data)
        
        vmax_error = np.max(abs_error)
        vmax_spread = np.max(ensemble_spread)
        
        # Create a figure for each timestep
        for t in timesteps:
            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot ground truth (top-left)
            im_truth = axes[0, 0].imshow(
                ground[t, feature_pos], 
                origin='lower', 
                cmap=cmap,
                vmin=vmin_field, 
                vmax=vmax_field
            )
            axes[0, 0].set_title('Ground Truth', fontweight='bold')
            
            # Add contour lines to ground truth
            contour_truth = axes[0, 0].contour(
                ground[t, feature_pos],
                colors='black',
                alpha=0.5,
                levels=10,
                linewidths=0.8
            )
            
            # Plot ensemble mean (top-right)
            im_mean = axes[0, 1].imshow(
                ensemble_mean[t], 
                origin='lower', 
                cmap=cmap,
                vmin=vmin_field, 
                vmax=vmax_field
            )
            axes[0, 1].set_title('Ensemble Mean', fontweight='bold')
            
            # Add contour lines to ensemble mean
            contour_mean = axes[0, 1].contour(
                ensemble_mean[t],
                colors='black',
                alpha=0.5,
                levels=10,
                linewidths=0.8
            )
            
            # Plot absolute error (bottom-left)
            im_error = axes[1, 0].imshow(
                abs_error[t], 
                origin='lower', 
                cmap='Reds',
                vmin=0, 
                vmax=vmax_error
            )
            axes[1, 0].set_title('Absolute Error', fontweight='bold')
            
            # Add contour lines to error
            contour_error = axes[1, 0].contour(
                abs_error[t],
                colors='black',
                alpha=0.5,
                levels=8,
                linewidths=0.8
            )
            
            # Plot ensemble spread (bottom-right)
            im_spread = axes[1, 1].imshow(
                ensemble_spread[t], 
                origin='lower', 
                cmap='plasma',
                vmin=0, 
                vmax=vmax_spread
            )
            axes[1, 1].set_title('Ensemble Spread', fontweight='bold')
            
            # Add contour lines to spread
            contour_spread = axes[1, 1].contour(
                ensemble_spread[t],
                colors='black',
                alpha=0.5,
                levels=8,
                linewidths=0.8
            )
            
            # Remove axis ticks
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Add colorbars
            cbar_truth = fig.colorbar(im_truth, ax=axes[0, 0], fraction=0.046, pad=0.04)
            cbar_mean = fig.colorbar(im_mean, ax=axes[0, 1], fraction=0.046, pad=0.04)
            cbar_error = fig.colorbar(im_error, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar_spread = fig.colorbar(im_spread, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            cbar_truth.set_label(display_feature_name, fontweight='bold')
            cbar_mean.set_label(display_feature_name, fontweight='bold')
            cbar_error.set_label('Absolute Error', fontweight='bold')
            cbar_spread.set_label('Spread', fontweight='bold')
            
            # Add overall title
            fig.suptitle(f'{display_feature_name} Forecast Comparison at Lead Time: {t*forecast_hours}h', 
                         fontweight='bold', fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure
            plt.savefig(os.path.join(save_dir, f'{feature_name}_t{t*forecast_hours}h.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{feature_name}_t{t*forecast_hours}h.pdf'), bbox_inches='tight')
            
            plt.close(fig)
        
        
        # Create a multi-timestep panel figure combining all lead times
        # For a more compact representation in publications
        if len(timesteps) <= 4:  # Only create this for a reasonable number of timesteps
            fig = plt.figure(figsize=(16, 4 * len(timesteps)))
            
            # Create a grid of subplots
            gs = fig.add_gridspec(len(timesteps), 4)
            
            # Plot each timestep and panel
            for i, t in enumerate(timesteps):
                # Ground truth
                ax1 = fig.add_subplot(gs[i, 0])
                im1 = ax1.imshow(ground[t, feature_pos], origin='lower', cmap=cmap, 
                                vmin=vmin_field, vmax=vmax_field)
                ax1.contour(ground[t, feature_pos], colors='black', alpha=0.5, levels=8, linewidths=0.8)
                ax1.set_xticks([])
                ax1.set_yticks([])
                if i == 0:
                    ax1.set_title('Ground Truth', fontweight='bold')
                ax1.set_ylabel(f'T+{t*forecast_hours}h', fontweight='bold')
                
                # Ensemble mean
                ax2 = fig.add_subplot(gs[i, 1])
                im2 = ax2.imshow(ensemble_mean[t], origin='lower', cmap=cmap, 
                                vmin=vmin_field, vmax=vmax_field)
                ax2.contour(ensemble_mean[t], colors='black', alpha=0.5, levels=8, linewidths=0.8)
                ax2.set_xticks([])
                ax2.set_yticks([])
                if i == 0:
                    ax2.set_title('Ensemble Mean', fontweight='bold')
                
                # Absolute error
                ax3 = fig.add_subplot(gs[i, 2])
                im3 = ax3.imshow(abs_error[t], origin='lower', cmap='Reds', 
                                vmin=0, vmax=vmax_error)
                ax3.contour(abs_error[t], colors='black', alpha=0.5, levels=8, linewidths=0.8)
                ax3.set_xticks([])
                ax3.set_yticks([])
                if i == 0:
                    ax3.set_title('Absolute Error', fontweight='bold')
                
                # Ensemble spread
                ax4 = fig.add_subplot(gs[i, 3])
                im4 = ax4.imshow(ensemble_spread[t], origin='lower', cmap='plasma', 
                                vmin=0, vmax=vmax_spread)
                ax4.contour(ensemble_spread[t], colors='black', alpha=0.5, levels=8, linewidths=0.8)
                ax4.set_xticks([])
                ax4.set_yticks([])
                if i == 0:
                    ax4.set_title('Ensemble Spread', fontweight='bold')
            
            # Add colorbars
            cbar_ax1 = fig.add_axes([0.905, 0.6, 0.02, 0.3])
            cbar_ax2 = fig.add_axes([0.905, 0.15, 0.02, 0.3])
            fig.colorbar(im1, cax=cbar_ax1).set_label(display_feature_name, fontweight='bold')
            fig.colorbar(im3, cax=cbar_ax2).set_label('Error/Spread', fontweight='bold')
            
            # Add title
            fig.suptitle(f'{display_feature_name} Forecast Evolution', fontweight='bold', fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            
            # Save the multi-panel figure
            plt.savefig(os.path.join(save_dir, f'{feature_name}_all_times.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{feature_name}_all_times.pdf'), bbox_inches='tight')
            
            plt.close(fig)
        

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