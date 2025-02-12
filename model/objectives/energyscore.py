import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

class EnergyScore:
    """Energy Score implementation in PyTorch.
    
    Computes Energy Score averaged over space and time. Uses weighted L2 norm and
    estimates spread using N-1 adjacent differences for memory efficiency.
    The score is computed separately for each tendency, level, and lag time.
    """
    
    def __init__(self):
        """Initialize Energy Score calculator. Assumes ensemble members in dimension 1."""
        pass
        
    def _compute_l2_norm(self, x1: torch.Tensor, x2: torch.Tensor, 
                        region: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute area-averaged L2 norm between two tensors."""
        # Compute squared differences
        diff = (x1 - x2) ** 2
        
        if region is not None:
            # Apply regional weighting
            diff = diff * region
            # Average over spatial dimensions (assuming last dimensions are spatial)
            spatial_dims = tuple(range(diff.ndim - 2, diff.ndim))
            norm = torch.sum(diff, dim=spatial_dims) / torch.sum(region, dim=spatial_dims)
        else:
            # Simple mean over spatial dimensions if no region weights provided
            spatial_dims = tuple(range(diff.ndim - 2, diff.ndim))
            norm = torch.mean(diff, dim=spatial_dims)
            
        # Take square root for L2 norm
        return torch.sqrt(norm)
    
    def compute_chunk(self, forecast: torch.Tensor, truth: torch.Tensor,
                     region: Optional[torch.Tensor] = None,
                     skipna: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Energy Score metrics for a chunk of data.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data with shape (time, n_members, n_vars, lat, lon)
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (spread, skill, score)
        """
        # Check minimum ensemble size
        n_members = forecast.shape[1]  # ensemble dimension is always 1
        if n_members < 2:
            raise ValueError("Need at least 2 ensemble members for Energy Score calculation")
            
        # Calculate ensemble mean
        forecast_mean = forecast.mean(dim=1)
        
        # Calculate spread using N-1 adjacent differences
        # E‖X - X'‖ ≈ (1 / (N-1)) Σₙ ‖X[n] - X[n+1]‖
        x1 = forecast[:, :-1]  # All members except last
        x2 = forecast[:, 1:]   # All members except first
        member_diffs = self._compute_l2_norm(x1, x2, region)
        spread = member_diffs.mean(dim=1)
        
        # Calculate skill (E‖X - Y‖)
        skill = self._compute_l2_norm(forecast_mean, truth, region)
        
        # Calculate Energy Score
        score = skill - 0.5 * spread
        
        if skipna:
            # Replace infinities with NaN
            spread = torch.where(torch.isinf(spread), torch.nan, spread)
            skill = torch.where(torch.isinf(skill), torch.nan, skill)
            score = torch.where(torch.isinf(score), torch.nan, score)
            
        return spread, skill, score
    
    def compute(self, forecast: torch.Tensor, truth: torch.Tensor,
                region: Optional[torch.Tensor] = None,
                skipna: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate Energy Score metrics on datasets with full temporal coverage.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data with shape (time, n_members, n_vars, lat, lon)
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Time-averaged (spread, skill, score)
        """
        # Compute metrics for the full dataset
        spread, skill, score = self.compute_chunk(forecast, truth, region, skipna)
        
        return spread, skill, score