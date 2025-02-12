import torch
import torch.nn as nn
from typing import Optional, Union

class EnsembleMeanRMSE:
    """Ensemble Mean RMSE implementation in PyTorch.
    
    Computes mean square error between ensemble mean and ground truth.
    Note that this has a bias of σ²/n where σ² is the ensemble variance
    and n is the ensemble size.
    """
    
    def __init__(self):
        """Initialize RMSE calculator. Assumes ensemble members in dimension 1."""
        pass
        
    def _compute_mse(self, forecast_mean: torch.Tensor, truth: torch.Tensor, 
                    region: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute area-averaged MSE between ensemble mean and truth."""
        # Compute squared differences
        diff = (forecast_mean - truth) ** 2  # (time, n_vars, lat, lon)
        
        # Always use fixed spatial dimensions (-2, -1) for lat, lon
        spatial_dims = (-2, -1)
        
        if region is not None:
            # Apply regional weighting
            diff = diff * region
            # Average over spatial dimensions
            mse = torch.sum(diff, dim=spatial_dims) / torch.sum(region, dim=spatial_dims)
        else:
            # Simple mean over spatial dimensions
            mse = torch.mean(diff, dim=spatial_dims)  # (time, n_vars)
            
        return mse
    
    def compute_chunk(self, forecast: torch.Tensor, truth: torch.Tensor,
                     region: Optional[torch.Tensor] = None,
                     skipna: bool = False) -> torch.Tensor:
        """
        Compute RMSE for a chunk of data.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data with shape (time, n_members, n_vars, lat, lon)
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            torch.Tensor: MSE values with shape (time, n_vars)
        """
        # Check minimum ensemble size
        n_members = forecast.shape[1]  # ensemble dimension is always 1
        if n_members < 2:
            raise ValueError("Need at least 2 ensemble members for Ensemble-Mean Score calculation")
            
        # Calculate ensemble mean
        forecast_mean = forecast.mean(dim=1)  # (time, n_vars, lat, lon)
        
        # Calculate MSE
        mse = self._compute_mse(forecast_mean, truth, region)  # (time, n_vars)
        
        # Convert MSE to RMSE by taking square root
        rmse = torch.sqrt(mse)
        
        if skipna:
            # Replace infinities with NaN
            rmse = torch.where(torch.isinf(rmse), torch.nan, rmse)
            
        return rmse
    
    def compute(self, forecast: torch.Tensor, truth: torch.Tensor,
                region: Optional[torch.Tensor] = None,
                skipna: bool = False) -> torch.Tensor:
        """
        Evaluate RMSE on datasets with full temporal coverage.
        
        Parameters:
            forecast (torch.Tensor): Ensemble forecast data with shape (time, n_members, n_vars, lat, lon)
            truth (torch.Tensor): Ground truth data with shape (time, n_vars, lat, lon)
            region (torch.Tensor, optional): Regional weights with shape (lat, lon)
            skipna (bool): Whether to skip NaN values
            
        Returns:
            torch.Tensor: MSE values with shape (time, n_vars)
        """
        # Compute metrics for the full dataset
        mse = self.compute_chunk(forecast, truth, region, skipna)
        
        return mse