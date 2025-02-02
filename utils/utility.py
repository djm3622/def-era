import os
import torch
import numpy as np

def validate_and_create_save_path(save_path, experiment_name):
    if experiment_name is None:
        raise ValueError("experiment_name must be set before validating save path")

    # Check if the path already exists
    if os.path.exists(save_path):
        raise AssertionError(f"Save path '{save_path}' already exists. Please choose a different experiment name.")

    # Create the directory and any necessary parent directories
    try:
        os.makedirs(save_path, exist_ok=False)
    except Exception as e:
        raise RuntimeError(f"Failed to create save directory: {str(e)}")
        
        
def descriptive_stats(imgs, ref):
    print(f'{torch.linalg.norm(imgs[0] - ref)}')
    
    print()
    
    print(f'Mean: {imgs.mean(dim=0).mean()}, Std: {imgs.mean(dim=0).std()}')
    
    print()
        
    for i in range(1, imgs.shape[0]):
        print(f'0:{i} - {torch.linalg.norm(imgs[0] - imgs[i])}')
        
    print()

    imgs_std = imgs.std(dim=0)[0]
    print(f'Mean: {imgs_std.mean()}, Median: {imgs_std.median()}')
    print(f'Max: {imgs_std.max()}, Min: {imgs_std.min()}')
    
    
def plot_images(imgs, ncols=None, figsize=(10, 10)):
    imgs = imgs.cpu().detach()
    num_images = imgs.shape[0]

    if ncols is None:
        ncols = int(num_images**0.5)  # Default to a square grid
    nrows = (num_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # If there's only one row or column, axes will not be a 2D array
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_images):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(imgs[i, 0])
        ax.axis('off')  # Hide axes

    # Hide any remaining empty subplots
    for i in range(num_images, nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_state_with_uncertainty(state, uncertainty, cmap='gray', uncertainty_alpha=0.5):    
    # Create a figure with a specific size
    plt.figure(figsize=(12, 6))
    
    # Create two subplots side by side
    plt.subplot(121)
    
    # Plot the state
    im1 = plt.imshow(state, cmap=cmap, origin='lower')
    plt.colorbar(im1, label='State Value')
    plt.title('State')
    
    # Plot the uncertainty
    plt.subplot(122)
    im2 = plt.imshow(state, cmap=cmap, origin='lower')
    
    # Overlay uncertainty with a different colormap and transparency
    uncertainty_normalized = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    im3 = plt.imshow(uncertainty_normalized, cmap='Reds', alpha=uncertainty_alpha, origin='lower')
    
    plt.colorbar(im3, label='Uncertainty')
    plt.title('State with Uncertainty Overlay')
    
    plt.tight_layout()
    
    
def standardize(imgs):
    return (imgs - imgs.mean(dim=(-2, -1), keepdim=True)) / imgs.std(dim=(-2, -1), keepdim=True)
