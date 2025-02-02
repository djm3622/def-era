import torch.optim as optim
import torch
from omegaconf import DictConfig
from torch.nn import Module

def get_adamw(
    model: Module,
    lr: float,
    cfg: DictConfig = {}
) -> optim.AdamW:
    
    return optim.AdamW(model.parameters(), lr=lr)