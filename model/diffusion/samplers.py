import torch
import numpy as np
from tqdm.notebook import tqdm
from diffusers import DPMSolverMultistepScheduler

# TODO
# file to hyperparam tune 
# DDIM, DPM++
# 25, 50, 100
# 0.9, 1.0, 1.1, 1.5, 2.5, 3.5, 4.5, 6.5

# if DDIM best -> eta = [0.0, 0.25, 0.5, 0.75, 1.0]


@torch.no_grad()
def sampling_with_cfg_ddim(
    model, operator, samples, t_timesteps, 
    alpha_bar, device, size, condition, 
    guidance_scale=7.5, num_steps=50, eta=0.0
):
    c, w, h = 1, size[0], size[1]
    imgs = torch.randn((samples, c, w, h), device=device)
    
    # Mixed conditioning setup (unchanged)
    b = condition.shape[0]
    operator_mask = torch.rand(b, device=condition.device) < 0.0  # Disabled
    operator_output = operator(condition, torch.ones(b, device=device), return_dict=False)[0].detach()
    mixed_condition = torch.where(operator_mask.view(-1,1,1,1), operator_output, condition)
    batch_cond = torch.cat([mixed_condition, torch.zeros_like(mixed_condition)])
    
    # DDIM timestep subset
    step_indices = torch.linspace(0, t_timesteps-1, num_steps, dtype=torch.long, device=device)
    
    for step in tqdm(step_indices.flip(0), desc="DDIM Sampling"):
        # Prepare timestep tensors
        timesteps = torch.full((samples,), step.item(), dtype=torch.int, device=device)
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
                
    return imgs, mixed_condition


@torch.no_grad()
def sampling_with_cfg_dpm(
    model, operator, samples, t_timesteps, beta, 
    alpha, alpha_bar, device, size, condition, 
    config, guidance_scale=7.5, num_steps=50, 
):
    c, w, h = 1, size[0], size[1]
    imgs = torch.randn((samples, c, w, h), device=device)
    
    # Create the mixed conditioning batch
    b = condition.shape[0]
    operator_mask = torch.rand(b, device=condition.device) < 0.0 # disable operator masking for now
    
    # Get operator output for the masked samples
    operator_output = operator(condition, torch.ones(b, device=condition.device), return_dict=False)[0].detach()
    
    # Create mixed condition batch
    mixed_condition = torch.where(
        operator_mask.view(-1, 1, 1, 1).expand_as(condition),
        operator_output,
        condition
    )    

    # Create full batch with conditioned and unconditioned
    batch_cond = torch.cat([mixed_condition, torch.zeros_like(mixed_condition)])
    
    # Initialize the DPM++ solver
    solver = DPMSolverMultistepScheduler(
        num_train_timesteps=t_timesteps,
        trained_betas=torch.linspace(1e-4, 0.02, t_timesteps),
        beta_schedule='linear',
        solver_order=config.solver_order,
        prediction_type=config.prediction_type,
        algorithm_type=config.algorithm_type
    )
    
    solver.set_timesteps(num_steps, device=device)
    
    for step in tqdm(solver.timesteps, desc="Denoising", unit="step"):
        # Create double-sized batch of images for parallel conditioned/unconditioned prediction
        double_imgs = torch.cat([imgs, imgs], dim=0)
        
        error = torch.randn_like(imgs) if step > 1 else torch.zeros_like(imgs)
        timesteps = torch.ones(samples, dtype=torch.int, device=device) * step
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
                        
    return imgs, mixed_condition


# class conditional sampling
@torch.no_grad()
def sampling_with_cfg_ddpm(
    model, operator, samples, t_timesteps, beta, alpha, 
    alpha_bar, device, size, condition, guidance_scale=7.5
):
    c, w, h = 1, size[0], size[1]
    imgs = torch.randn((samples, c, w, h), device=device)
    
    # Create the mixed conditioning batch
    b = condition.shape[0]
    operator_mask = torch.rand(b, device=condition.device) < 0.0 # disable operator masking for now
    
    # Get operator output for the masked samples
    operator_output = operator(condition, torch.ones(b, device=condition.device), return_dict=False)[0].detach()
    
    # Create mixed condition batch
    mixed_condition = torch.where(
        operator_mask.view(-1, 1, 1, 1).expand_as(condition),
        operator_output,
        condition
    )    

    # Create full batch with conditioned and unconditioned
    batch_cond = torch.cat([mixed_condition, torch.zeros_like(mixed_condition)])    
    
    for step in tqdm(range(t_timesteps-1, -1, -1), desc="Denoising", unit="step"):
        # Create double-sized batch of images for parallel conditioned/unconditioned prediction
        double_imgs = torch.cat([imgs, imgs], dim=0)
        
        error = torch.randn_like(imgs) if step > 1 else torch.zeros_like(imgs)
        timesteps = torch.ones(samples, dtype=torch.int, device=device) * step
        double_timesteps = torch.concat([timesteps, timesteps], dim=0)
        
        # Get parameters for current timestep
        beta_t = beta[timesteps].view(samples, 1, 1, 1)
        alpha_t = alpha[timesteps].view(samples, 1, 1, 1)
        alpha_bar_t = alpha_bar[timesteps].view(samples, 1, 1, 1)
        
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
                
    return imgs, mixed_condition