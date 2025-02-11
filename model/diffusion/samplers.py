import torch
import numpy as np
from typing import Tuple
from omegaconf import DictConfig
from tqdm.auto import tqdm  
from torch.nn import Module
from diffusers import DPMSolverMultistepScheduler


def standardize(
    out: torch.Tensor
) -> torch.Tensor:    
    
    mu = out.mean(dim=(-2, -1), keepdims=True)
    std = out.std(dim=(-2, -1), keepdims=True)
    
    return (out - mu) / std


@torch.no_grad()
def sampling_with_cfg_ddim(
    model: Module, 
    t_timesteps: int, 
    beta: torch.Tensor, 
    alpha: torch.Tensor, 
    alpha_bar: torch.Tensor, 
    device: torch.device, 
    condition: torch.Tensor, 
    guidance_scale: float = 7.5, 
    num_steps: int = 50, 
    eta: float = 0.0,
    corrector: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    imgs = torch.randn_like(condition, device=device)
    batch_cond = torch.cat([condition, torch.zeros_like(condition)])
    
    # DDIM timestep subset
    step_indices = torch.linspace(0, t_timesteps-1, num_steps, dtype=torch.long, device=device)
    
    sample_bar = tqdm(
        step_indices.flip(0), 
        desc='DDIM', 
        leave=False,
        mininterval=1.0
    )
    
    for step in sample_bar:
        # Prepare timestep tensors
        timesteps = torch.full((condition.shape[0],), step.item(), dtype=torch.int, device=device)
        double_imgs = torch.cat([imgs, imgs], dim=0)
        double_timesteps = torch.cat([timesteps, timesteps], dim=0)
        
        # Get alpha_bar values
        alpha_bar_t = alpha_bar[step].view(1, 1, 1, 1)
        prev_step = step_indices[step_indices < step].max() if step > 0 else torch.tensor(0, device=device)
        alpha_bar_prev = alpha_bar[prev_step].view(1, 1, 1, 1)
        
        # Predict noise with CFG
        double_next_state = torch.cat([batch_cond, double_imgs], dim=1)
        noise_pred = model(double_next_state, double_timesteps, return_dict=False)[0]
        noise_pred_cond, noise_pred_uncond = torch.chunk(noise_pred, 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # DDIM update with eta-controlled stochasticity
        predicted_x0 = (imgs - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        variance = (1 - alpha_bar_prev)/(1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev)
        variance = torch.clamp(variance, min=0)
        sigma_t = eta * torch.sqrt(variance)
        
        noise = torch.randn_like(imgs) if eta > 0 and step > 1 else torch.zeros_like(imgs)
        imgs = (
            alpha_bar_prev.sqrt() * predicted_x0 + 
            torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * noise_pred + 
            sigma_t * noise
        )
        
        # corrector
        if corrector:
            imgs = standardize(imgs)
                
    return imgs, condition


@torch.no_grad()
def sampling_with_cfg_dpm(
    model: Module, 
    solver: DPMSolverMultistepScheduler, 
    t_timesteps: int, 
    beta: torch.Tensor, 
    alpha: torch.Tensor, 
    alpha_bar: torch.Tensor, 
    device: torch.device, 
    condition: torch.Tensor, 
    guidance_scale: float = 7.5, 
    num_steps: float = 25, 
    corrector: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    imgs = torch.randn_like(condition, device=device)
    batch_cond = torch.cat([condition, torch.zeros_like(condition)])
    
    solver.set_timesteps(num_steps, device=device)
    
    sample_bar = tqdm(
        solver.timesteps, 
        desc='DPM++', 
        leave=False,
        mininterval=1.0
    )
    
    for step in sample_bar:
        # Create double-sized batch of images for parallel conditioned/unconditioned prediction
        double_imgs = torch.cat([imgs, imgs], dim=0)
        
        error = torch.randn_like(imgs) if step > 1 else torch.zeros_like(imgs)
        timesteps = torch.ones(condtion.shape[0], dtype=torch.int, device=device) * step
        double_timesteps = torch.concat([timesteps, timesteps], dim=0)
        
        # Get both conditioned and unconditioned predictions
        double_next_state = torch.cat([batch_cond, double_imgs], dim=1)
        noise_pred = model(double_next_state, double_timesteps, return_dict=False)[0]
        
        # Split predictions
        noise_pred_cond, noise_pred_uncond = torch.chunk(noise_pred, 2)
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Use the DPM++ solver to predict the next state
        imgs = solver.step(noise_pred, step, imgs).prev_sample
        
        # corrector
        if corrector:
            imgs = standardize(imgs)
                        
    return imgs, condition


@torch.no_grad()
def sampling_with_cfg_ddpm(
    model: Module, 
    t_timesteps: int, 
    beta: torch.Tensor, 
    alpha: torch.Tensor, 
    alpha_bar: torch.Tensor, 
    device: torch.device, 
    condition: torch.Tensor, 
    guidance_scale: int = 7.5,
    corrector: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    imgs = torch.randn_like(condition, device=device)
    batch_cond = torch.cat([condition, torch.zeros_like(condition)])    
    
    sample_bar = tqdm(
        range(t_timesteps-1, -1, -1), 
        desc='DDPM',
        leave=False,
        mininterval=1.0
    )
    
    for step in sample_bar:
        # Create double-sized batch of images for parallel conditioned/unconditioned prediction
        double_imgs = torch.cat([imgs, imgs], dim=0)
        
        error = torch.randn_like(imgs) if step > 1 else torch.zeros_like(imgs)
        timesteps = torch.ones(condition.shape[0], dtype=torch.int, device=device) * step
        double_timesteps = torch.concat([timesteps, timesteps], dim=0)
        
        # Get parameters for current timestep
        beta_t = beta[timesteps].view(condition.shape[0], 1, 1, 1)
        alpha_t = alpha[timesteps].view(condition.shape[0], 1, 1, 1)
        alpha_bar_t = alpha_bar[timesteps].view(condition.shape[0], 1, 1, 1)
        
        # Get both conditioned and unconditioned predictions
        double_next_state = torch.cat([batch_cond, double_imgs], dim=1)
        noise_pred = model(double_next_state, double_timesteps, return_dict=False)[0]
        
        # Split predictions
        noise_pred_cond, noise_pred_uncond = torch.chunk(noise_pred, 2)
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Apply diffusion formula
        mu = 1 / torch.sqrt(alpha_t) * (imgs - ((beta_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        sigma = torch.sqrt(beta_t)
        imgs = mu + sigma * error
        
        # corrector
        if corrector:
            imgs = standardize(imgs)
                
    return imgs, condition