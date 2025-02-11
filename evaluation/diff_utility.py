import torch
from typing import Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from torch.utils.data import Dataset
from torch.nn import Module

from diffusers import DPMSolverMultistepScheduler


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

    return None