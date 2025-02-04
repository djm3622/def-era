from diffusers import UNet2DModel
import torch
from torch import nn
from omegaconf import DictConfig


def get_unet_based_model(
    x: int, y: int,
    channels: int,
    cfg: DictConfig = {}
) -> UNet2DModel :
    
    return UNet2DModel(
        sample_size=(x, y),        
        in_channels=channels*2,  # accounting for the condition      
        out_channels=channels,         
        layers_per_block=4,      
        block_out_channels=(256, 384, 512, 1024),  
        down_block_types=(
            "DownBlock2D",      # 128 channels at 32x64
            "AttnDownBlock2D",  # 256 channels at 16x32
            "AttnDownBlock2D",  # 384 channels at 8x16
            "AttnDownBlock2D",  # 512 channels at 4x8
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        )
    )