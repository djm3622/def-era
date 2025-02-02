import torch.optim as optim
import torch
from omegaconf import DictConfig
from torch.nn import Module

def get_onecycle_lr(
    optimizer: optim.Optimizer,
    max_lr: float,
    epoches: int,
    steps_per_epoch: int,
    pct_start: float = 0.1,
    div_factor: int = 25,
    final_div_factor: float = 1e4,
    cfg: DictConfig = {}
) -> optim.lr_scheduler:
    
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epoches,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,  
        div_factor=div_factor,  
        final_div_factor=final_div_factor 
    )