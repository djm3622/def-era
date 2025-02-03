import torch
import torch.nn as nn
from torch.nn import Module


def get_diffusion_loss() -> Module:
    return nn.MSELoss()