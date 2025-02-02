from diffusers import UNet2DModel
import torch
from torch import nn
from omegaconf import DictConfig


def get_unet_based_model(
    x: int, y: int,
    in_channels: int,
    out_channels: int,
    cfg: DictConfig = {}
) -> UNet2DModel :
    
    return UNet2DModel(
        sample_size=(cfg, cfg),        
        in_channels=in_channels,         
        out_channels=out_channels,         
        layers_per_block=4,      
        block_out_channels=(128, 128, 256, 256),  
        down_block_types=(
            "DownBlock2D",      # 32 channels at 32x64
            "AttnDownBlock2D",  # 64 channels at 16x32
            "AttnDownBlock2D",  # 128 channels at 8x16
            "AttnDownBlock2D",  # 512 channels at 4x8
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        )
    )