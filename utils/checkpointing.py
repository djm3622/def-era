"""Utilities for model checkpointing."""

from typing import Dict, Any
import torch


def load_checkpoint(
    filename: str,
    litmodel: torch.nn.Module,
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional
    Returns:
        Litghtning object containing the model and all objects declared within
    """

    return litmodel.load_from_checkpoint(filename)
