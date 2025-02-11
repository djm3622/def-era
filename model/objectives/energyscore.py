import torch
import torch.nn as nn
from typing import Optional, Union, Tuple


class EnergyScore:
    """Energy Score implementation in PyTorch.
    
    Computes Energy Score averaged over space and time. Uses weighted L2 norm and
    estimates spread using N-1 adjacent differences for memory efficiency.
    The score is computed separately for each tendency, level, and lag time.
    
    Parameters:
        ensemble_dim (str): Dimension name for ensemble members, defaults to 'realization'
    """
    
    def __init__(self, ensemble_dim: str = 'realization'):
        self.ensemble_dim = ensemble_dim
        
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
                where:
                - time: number of forecast start times
                - n_members: number of ensemble members (realization dimension)
                - n_vars: number of variables/tendencies being forecast
                - lat, lon: spatial dimensions for the grid
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
                Same as forecast but without the ensemble dimension
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (spread, skill, score)
        """
        # Get ensemble dimension index
        ens_dim = forecast.names.index(self.ensemble_dim) if forecast.names else 0
        
        # Check minimum ensemble size
        n_members = forecast.shape[ens_dim]
        if n_members < 2:
            raise ValueError("Need at least 2 ensemble members for Energy Score calculation")
            
        # Calculate ensemble mean
        forecast_mean = forecast.mean(dim=ens_dim)
        
        # Calculate spread using N-1 adjacent differences
        # E‖X - X'‖ ≈ (1 / (N-1)) Σₙ ‖X[n] - X[n+1]‖
        x1 = torch.index_select(forecast, ens_dim, torch.arange(n_members-1))
        x2 = torch.index_select(forecast, ens_dim, torch.arange(1, n_members))
        member_diffs = self._compute_l2_norm(x1, x2, region)
        spread = torch.mean(member_diffs, dim=ens_dim)
        
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
        
        # Average over time dimension (assumed to be first dimension after removing ensemble)
        time_dim = 0
        spread = torch.mean(spread, dim=time_dim)
        skill = torch.mean(skill, dim=time_dim)
        score = torch.mean(score, dim=time_dim)
        
        return spread, skill, score