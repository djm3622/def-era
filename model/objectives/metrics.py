import torch
import numpy as np
import scipy as sp


def crps(ensemble, obs):
    """
    ensemble: Tensor of shape (n_members, n_samples)
    obs: Tensor of shape (n_samples,)
    """
    # Term 1: Mean absolute error between members and observation
    term1 = torch.mean(torch.abs(ensemble - obs), dim=0)
    
    # Term 2: Mean absolute difference between all member pairs
    diff = torch.abs(ensemble.unsqueeze(1) - ensemble.unsqueeze(0))  # (n_members, n_members, n_samples)
    term2 = torch.mean(diff, dim=(0, 1)) / 2
    
    return torch.mean(term1 - term2)  # Average over all samples


def brier_score(ensemble, obs, threshold=10.0):
    """
    ensemble: Array of shape (n_members, n_samples)
    obs: Array of shape (n_samples,)
    """
    # Compute forecast probability
    event = (ensemble > threshold).mean(dim=0)  # (n_samples,)
    
    # Observed binary outcome
    obs_binary = (obs > threshold).to(torch.float)
    
    return torch.mean((event - obs_binary) ** 2)


# convert to numpy before sending in
def ignorance_score(ensemble, obs):
    """
    ensemble: Array of shape (n_members, n_samples)
    obs: Array of shape (n_samples,)
    """
    n_samples = obs.shape[0]
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Fit KDE to ensemble members for sample i
        kde = sp.stats.gaussian_kde(ensemble[:, i])
        scores[i] = -kde.logpdf(obs[i])
    
    return np.mean(scores)  # Average over samples


def spread_skill(ensemble, obs):
    """
    ensemble: Array of shape (n_members, n_samples)
    obs: Array of shape (n_samples,)
    """
    ensemble_mean = torch.mean(ensemble, dim=0)
    spread = torch.std(ensemble, dim=0)  # Spread at each sample
    skill = torch.abs(ensemble_mean - obs)  # RMSE component
    
    # Average over all samples
    return torch.mean(spread), torch.mean(skill)