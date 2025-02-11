import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

class CRPS:
    """Continuous Ranked Probability Score implementation in PyTorch.
    
    Computes CRPS averaged over space and time using the PWM method from Zamo & Naveau, 2018.
    The score is computed separately for each tendency, level, and lag time.
    
    Parameters:
        ensemble_dim (str): Dimension name for ensemble members, defaults to 'realization'
    """
    
    def __init__(self, ensemble_dim: str = 'realization'):
        self.ensemble_dim = ensemble_dim
        
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
                where:
                - time: number of forecast start times
                - n_members: number of ensemble members (realization dimension)
                - n_vars: number of variables/tendencies being forecast
                - lat, lon: spatial dimensions for the grid
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
                Same as forecast but without the ensemble dimension
            region (torch.Tensor, optional): Regional weights
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (spread, skill, score)
        """
        # Get ensemble dimension index
        ens_dim = forecast.names.index(self.ensemble_dim) if forecast.names else 0
        
        # Split ensemble into pairs for spread calculation
        n_members = forecast.shape[ens_dim]
        if n_members < 2:
            raise ValueError("Need at least 2 ensemble members for CRPS calculation")
            
        # Calculate ensemble mean using PWM method
        forecast_mean = forecast.mean(dim=ens_dim)
        
        # Calculate spread (E|X - X'|)
        # Randomly pair ensemble members
        idx = torch.randperm(n_members)
        x1 = torch.index_select(forecast, ens_dim, idx[:n_members//2])
        x2 = torch.index_select(forecast, ens_dim, idx[n_members//2:n_members//2*2])
        spread = self._compute_absolute_differences(x1, x2, region)
        
        # Calculate skill (E|X - Y|)
        skill = self._compute_absolute_differences(forecast_mean, truth, region)
        
        # Calculate CRPS score
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
        Evaluate CRPS metrics on datasets with full temporal coverage.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data
            truth (torch.Tensor): Ground truth data
            region (torch.Tensor, optional): Regional weights
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Time-averaged (spread, skill, score)
        """
        # Compute metrics for the full dataset
        spread, skill, score = self.compute_chunk(forecast, truth, region, skipna)
        
        # Average over time dimension (assumed to be first dimension after removing ensemble)
        time_dim = 0
        spread = torch.mean(spread, dim=time_dim)
        skill = torch.mean(skill, dim=time_dim)
        score = torch.mean(score, dim=time_dim)
        
        return spread, skill, score
    