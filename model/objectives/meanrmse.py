import torch
import torch.nn as nn
from typing import Optional, Union

class EnsembleMeanRMSE:
    """Ensemble Mean RMSE implementation in PyTorch.
    
    Computes mean square error between ensemble mean and ground truth.
    Note that this has a bias of σ²/n where σ² is the ensemble variance
    and n is the ensemble size.
    
    Parameters:
        ensemble_dim (str): Dimension name for ensemble members, defaults to 'realization'
    """
    
    def __init__(self, ensemble_dim: str = 'realization'):
        self.ensemble_dim = ensemble_dim
        
    def _compute_mse(self, forecast_mean: torch.Tensor, truth: torch.Tensor, 
                    region: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute area-averaged MSE between ensemble mean and truth."""
        # Compute squared differences
        diff = (forecast_mean - truth) ** 2
        
        if region is not None:
            # Apply regional weighting
            diff = diff * region
            # Average over spatial dimensions (assuming last dimensions are spatial)
            spatial_dims = tuple(range(diff.ndim - 2, diff.ndim))
            mse = torch.sum(diff, dim=spatial_dims) / torch.sum(region, dim=spatial_dims)
        else:
            # Simple mean over spatial dimensions if no region weights provided
            spatial_dims = tuple(range(diff.ndim - 2, diff.ndim))
            mse = torch.mean(diff, dim=spatial_dims)
            
        return mse
    
    def compute_chunk(self, forecast: torch.Tensor, truth: torch.Tensor,
                     region: Optional[torch.Tensor] = None,
                     skipna: bool = False) -> torch.Tensor:
        """
        Compute RMSE for a chunk of data.
        
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
            torch.Tensor: MSE values
        """
        # Get ensemble dimension index
        ens_dim = forecast.names.index(self.ensemble_dim) if forecast.names else 0
        
        # Calculate ensemble mean
        forecast_mean = forecast.mean(dim=ens_dim)
        
        # Calculate MSE
        mse = self._compute_mse(forecast_mean, truth, region)
        
        if skipna:
            # Replace infinities with NaN
            mse = torch.where(torch.isinf(mse), torch.nan, mse)
            
        return mse
    
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
            torch.Tensor: Time-averaged MSE values
        """
        # Compute metrics for the full dataset
        mse = self.compute_chunk(forecast, truth, region, skipna)
        
        # Average over time dimension (assumed to be first dimension after removing ensemble)
        time_dim = 0
        mse = torch.mean(mse, dim=time_dim)
        
        return mse