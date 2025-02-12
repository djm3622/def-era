import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

class CRPS:
    """Continuous Ranked Probability Score implementation in PyTorch.
    
    Computes CRPS averaged over space using the PWM method from Zamo & Naveau, 2018.
    The score is computed separately for each time step, tendency, level, and lag time.
    """
    
    def __init__(self):
        """Initialize CRPS calculator. Assumes ensemble members in dimension 1."""
        pass
        
    def _compute_absolute_differences(self, x1: torch.Tensor, x2: torch.Tensor, 
                                    region: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute area-averaged L1 norm between two tensors."""
        diff = torch.abs(x1 - x2)
        
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
            
        return norm
    
    def compute_chunk(self, forecast: torch.Tensor, truth: torch.Tensor,
                     region: Optional[torch.Tensor] = None,
                     skipna: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute CRPS metrics for a chunk of data.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data with shape (time, n_members, n_vars, lat, lon)
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (spread, skill, score) each with shape (time, n_vars)
        """
        # Check minimum ensemble size
        n_members = forecast.shape[1]  # ensemble dimension is always 1
        if n_members < 2:
            raise ValueError("Need at least 2 ensemble members for CRPS Score calculation")
            
        # Calculate ensemble mean
        forecast_mean = forecast.mean(dim=1)  # (time, n_vars, lat, lon)
        
        # Calculate spread (E|X - X'|)
        # Randomly pair ensemble members
        idx = torch.randperm(n_members)
        n_pairs = n_members // 2
        x1 = forecast[:, idx[:n_pairs]]  # (time, n_pairs, n_vars, lat, lon)
        x2 = forecast[:, idx[n_pairs:n_pairs*2]]  # (time, n_pairs, n_vars, lat, lon)
        
        # Compute differences and mean over pairs
        member_diffs = self._compute_absolute_differences(x1, x2, region)  # (time, n_pairs, n_vars)
        spread = member_diffs.mean(dim=1)  # (time, n_vars)
        
        # Calculate skill (E|X - Y|)
        skill = self._compute_absolute_differences(forecast_mean, truth, region)  # (time, n_vars)
        
        # Calculate CRPS score
        score = skill - 0.5 * spread  # (time, n_vars)
        
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
        Evaluate CRPS metrics on datasets with full temporal coverage.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data with shape (time, n_members, n_vars, lat, lon)
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (spread, skill, score) each with shape (time, n_vars)
        """
        # Compute metrics for the full dataset
        spread, skill, score = self.compute_chunk(forecast, truth, region, skipna)
        
        return spread, skill, score