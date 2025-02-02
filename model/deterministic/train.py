import torch
import torch.optim as optim
from tqdm.auto import tqdm  
import time

from omegaconf import DictConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.optim as optim
import model.utility as utility
import utils.wandb_helper as wandb_helper


# generate a trajecory and log the ground truth (can't log ground truth because we don't know)
def sample():
    pass


def step(
    batch: tuple, 
    model: Module, 
    criterion: Module
) -> torch.Tensor:
    
    x, y, t = batch
    out = model(x, t.squeeze(-1), return_dict=False)[0]
    loss = criterion(out, y)

    return loss
    

def training_loop(
    accelerator: Accelerator, 
    train: DataLoader, 
    valid: DataLoader, 
    model: Module, 
    epochs: int, 
    patience: int, 
    criterion: Module, 
    save_path: str, 
    optimizer: optim, 
    scheduler: optim.lr_scheduler, 
    val_delay: int = 1, 
    loading_bar: bool = False,
    config: DictConfig = {}
) -> None:    
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Debug prints
    accelerator.print(f"Rank: {accelerator.process_index}")
    accelerator.print(f"Train dataset size: {len(train.dataset)}")
    accelerator.print(f"Batch size: {train.batch_size}")
    accelerator.print(f"Number of workers: {train.num_workers if hasattr(train, 'num_workers') else 'N/A'}")
    
    start_time = time.time()
    first_batch = next(iter(train))
    fetch_time = time.time() - start_time
    accelerator.print(f"Time to fetch first batch: {fetch_time:.2f} seconds")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training batch progress bar
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
                train_bar.set_postfix(train_loss=loss.item())
                                            
        train_loss /= len(train)
        gathered_train_loss = accelerator.gather(torch.tensor([train_loss]).to(accelerator.device)).mean().item()
        
        # quit early if no validation to do
        if epoch % val_delay != 0:
            accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}')
            
            # Log just train loss
            if accelerator.is_main_process:
                wandb_helper.log_losses(
                    train_loss=gathered_train_loss,
                    valid_loss=None,
                    step=epoch
                )
                
            accelerator.wait_for_everyone()
            
            utility.save_training_state(
                accelerator, epoch, model, 
                optimizer, scheduler, save_path
            )
            
            continue
            
        model.eval()
        val_loss = 0
        
        # Validation batch progress bar
        valid_bar = tqdm(
            valid, 
            desc=f'Validation', 
            leave=False,
            disable=not (loading_bar and accelerator.is_main_process),
            mininterval=1.0
        )
        
        with torch.no_grad():
            for valid_batch in valid_bar:
                loss = step(valid_batch, model, criterion)
                val_loss += loss.item()
                
                if loading_bar:
                    loader.set_postfix(val_loss=loss.item())
                                            
        val_loss /= len(valid)
        accelerator.wait_for_everyone()
        gathered_val_loss = accelerator.gather(torch.tensor([val_loss]).to(accelerator.device)).mean().item()
        
        accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}, Validation Loss: {gathered_val_loss}')

        
        # Log epoch metrics to wandb if main process
        if accelerator.is_main_process:
            wandb_helper.log_losses(
                train_loss=gathered_train_loss,
                valid_loss=gathered_val_loss,
                epoch=epoch
            )
                
        accelerator.wait_for_everyone()
            
        utility.save_training_state(
            accelerator, epoch, model, 
            optimizer, scheduler, save_path
        )
        
        
        