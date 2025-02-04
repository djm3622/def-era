import torch
import torch.optim as optim
from tqdm.notebook import tqdm
from wandb_helper import log_losses
import random
import torch
import numpy as np
import os
from typing import Tuple
from omegaconf import DictConfig

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.optim as optim


# class conditional sampling
@torch.no_grad()
def sampling_with_cfg(
    model: Module, 
    operator: Module, 
    samples: int, 
    t_timesteps: int, 
    beta: torch.Tensor, 
    alpha: torch.Tensor, 
    alpha_bar: torch.Tensor, 
    accelerator: Accelerator, 
    size: int, 
    condition : torch.Tensor, 
    force_constants: torch.Tensor,
    guidance_scale: int = 7.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # TODO: VERIFY THAT I AM ADDING THE FORCINGS CORRECT
    
    c, w, h = 1, size[0], size[1]
    imgs = torch.randn((samples, c, w, h), device=accelerator.device)
    
    # Create the mixed conditioning batch
    b = condition.shape[0]
    operator_mask = torch.rand(b, device=condition.device) < 0.5
    
    # Get operator output for the masked samples
    operator_output = operator(
        torch.cat([condition, force_constants], dim=1),  # add forcings and constansts. VERIFY THIS WORKS 
        torch.ones(b, device=condition.device), 
        return_dict=False
    )[0].detach()
    
    # Create mixed condition batch
    mixed_condition = torch.where(
        operator_mask.view(-1, 1, 1, 1).expand_as(condition),
        operator_output,
        condition
    )    

    # Create full batch with conditioned and unconditioned
    batch_cond = torch.cat([mixed_condition, torch.zeros_like(mixed_condition)])    
    
    for step in range(t_timesteps-1, -1, -1):
        # Create double-sized batch of images for parallel conditioned/unconditioned prediction
        double_imgs = torch.cat([imgs, imgs], dim=0)
        
        error = torch.randn_like(imgs) if step > 1 else torch.zeros_like(imgs)
        timesteps = torch.ones(samples, dtype=torch.int, device=accelerator.device) * step
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


# forcings are passed at each step. THIS IS NOT A CONDITION, it is simply additional information
# this may be changed depending results obtained from training
def step(
    train_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
    model: Module, 
    criterion: Module
) -> torch.Tensor:
    
    # TODO: VERIFY THAT I AM ADDING THE FORCINGS CORRECT
    
    clean_images, force_constants, noisy_images, rand_timesteps = train_batch
    alpha_bar_t = alpha_bar[rand_timesteps].view(-1, 1, 1, 1)

    b, c, h, w = clean_images.shape
    
    null_mask = torch.rand(b, device=clean_images.device) < 0.1

    # For non-null samples, decide between clean images and operator output (50% chance each)
    operator_mask = torch.rand(b, device=clean_images.device) < 0.5
    operator_mask = operator_mask & ~null_mask  # Only apply to non-null samples

    # Apply operator to clean images where needed
    with torch.no_grad():
        operator_output = operator(
            torch.cat([clean_images, force_constants], dim=1),  # add forcings along channel dim VERIFY THIS WORKS
            torch.ones(b, device=clean_images.device), 
            return_dict=False)[0].detach() if operator_mask.any() else clean_images
        
    # Create target images - use operator output where appropriate
    target_images = torch.where(
        operator_mask.view(-1, 1, 1, 1).expand_as(clean_images),
        operator_output,
        clean_images
    )
    
    # Create the final conditioning
    mixed_condition = torch.zeros_like(clean_images)  # Start with zeros (null conditioning)
    # Add clean images where appropriate
    mixed_condition = torch.where(
        (~null_mask & ~operator_mask).view(-1, 1, 1, 1).expand_as(clean_images),
        clean_images,
        mixed_condition
    )
    
    # Add operator output where appropriate
    mixed_condition = torch.where(
        operator_mask.view(-1, 1, 1, 1).expand_as(clean_images),
        operator_output,
        mixed_condition
    )

    # Calculate next state using the appropriate target images
    next_state = (torch.sqrt(alpha_bar_t) * target_images) + (torch.sqrt(1 - alpha_bar_t) * noisy_images)

    model_inpt = torch.concat([mixed_condition, next_state], dim=1)
    noise_pred = model(model_inpt, rand_timesteps.squeeze(-1), return_dict=False)[0]

    return criterion(noise_pred, noisy_images)


def training_loop(
    accelerator: Accelerator, 
    train: DataLoader, 
    valid: DataLoader, 
    model: Module, 
    operator: Module, 
    epochs: int, 
    criterion: Module, 
    save_path: str, 
    optimizer: optim, 
    scheduler: optim.lr_scheduler, 
    t_timesteps: int, 
    val_delay: int = 1, 
    loading_bar: bool = False,
    config: DictConfig = {}
) -> None:
    
    # Sanity Check
    accelerator.print(f"Rank: {accelerator.process_index}")
    accelerator.print(f"Train dataset size: {len(train.dataset)}")
    accelerator.print(f"Number of workers: {train.num_workers if hasattr(train, 'num_workers') else 'N/A'}")
    
    start_time = time.time()
    first_batch = next(iter(train))
    fetch_time = time.time() - start_time
    accelerator.print(f"Time to fetch first batch: {fetch_time:.2f} seconds")
    
    # allocate memory for these
    beta = torch.linspace(1e-4, 0.02, t_timesteps).to(accelerator.device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(accelerator.device)
    
    # 0.10 chance to drop conditioning
    # 0.50 to use the operator output
    # use (states + operator output) as as next_state prediction
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        
        train_bar = tqdm(
            train, 
            desc=f'Training', 
            leave=False,
            disable=not (loading_bar and accelerator.is_main_process),
            mininterval=1.0  # Update more frequently
        )
        
        for batch_idx, train_batch in enumerate(train_bar):
            with accelerator.accumulate(model):
                loss = step(train_batch, model, criterion)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                                
            if loading_bar:
                train_bar.set_postfix(
                    train_loss=loss.item(), 
                    lr=scheduler.get_last_lr()[0]
                )
                                                            
        train_loss /= len(train)
        gathered_train_loss = accelerator.gather(torch.tensor([train_loss]).to(accelerator.device)).mean().item()
        accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}')
        
        samples_np, conditions_np = None, None

        # monitor the quality of the samples
        if epoch % sample_delay == 0:
            # Get a batch from validation set
            valid_batch = next(iter(valid))
            valid_images, force_constants, _, _ = valid_batch

            # Generate samples and get mixed conditions
            samples, mixed_condition = sampling_with_cfg(
                model=model,
                samples=valid_images.shape[0],  # Use same batch size as validation
                t_timesteps=t_timesteps,
                beta=beta,
                alpha=alpha,
                alpha_bar=alpha_bar,
                accelerator=accelerator,
                size=(valid_images.shape[-2], valid_images.shape[-1]),
                condition=valid_images,
                operator=operator,
                force_constants=force_constants,
                guidance_scale=7.5
            )

            # Convert both samples and conditions to numpy for logging
            samples_np = samples.cpu().numpy()
            conditions_np = mixed_condition.cpu().numpy()

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            log_losses(
                train_loss=gathered_train_loss,
                valid_loss=None,
                step=epoch,
                samples=samples_np,
                conditions=conditions_np
            )
            
        accelerator.wait_for_everyone()
            
        save_training_state(
            accelerator=accelerator,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng_states=None,  # Will be collected inside the function
            output_dir=save_path+'states/'
        )
            
        accelerator.wait_for_everyone()