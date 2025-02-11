import torch
from typing import Dict, Callable, Tuple
from omegaconf import DictConfig
from diffusers import DPMSolverMultistepScheduler
from . import samplers
from torch.nn import Module


def get_sampler(
    sampler_type: str,
    model: Module, 
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
    cfg: DictConfig
) -> Tuple[Dict, Callable]:
        
    return sampler_lookup[sampler_type](
        cfg, model, beta, alpha, alpha_bar, device
    )


def get_ddpm(
    cfg: DictConfig,
    model: Module, 
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict, Callable]:
    
    # none is used when param should be passed at runtime
    params = {
        'model': model, 
        't_timesteps': cfg['dataset']['timestep'],
        'beta': beta,
        'alpha': alpha, 
        'alpha_bar': alpha_bar, 
        'device': device, 
        'condition': None, 
        'guidance_scale': cfg['evaluation']['guidance_scale'],
        'corrector': cfg['evaluation']['corrector']
    }
    sampler = samplers.sampling_with_cfg_ddpm
    
    return (params, sampler)


# only applicable to the DPM++ solver
def get_solver(
    num_train_timesteps: int,
    trained_betas: torch.Tensor,
    solver_order: int = 2,
    prediction_type: str = 'epsilon',
    algorithm_type: str = 'dpmsolver++',
    config: DictConfig = {}
    
) -> DPMSolverMultistepScheduler:
    
    return DPMSolverMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        trained_betas=trained_betas,
        solver_order=solver_order,
        prediction_type=prediction_type,
        algorithm_type=algorithm_type
    )


def get_dpm(
    cfg: DictConfig,
    model: Module, 
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict, Callable]:
    
    # none is used when param should be passed at runtime
    params = {
        'model': model, 
        'solver': get_solver(
            num_train_timesteps = cfg['dataset']['timestep'],
            trained_betas = beta.clone().detach(),
            solver_order = cfg['evaluation']['solver_order'],
            prediction_type = cfg['evaluation']['prediction_type'],
            algorithm_type = cfg['evaluation']['algorithm_type']
        ), 
        't_timesteps': cfg['dataset']['timestep'], 
        'beta': beta, 
        'alpha': alpha, 
        'alpha_bar': alpha_bar, 
        'device': device, 
        'condition': None, 
        'guidance_scale': cfg['evaluation']['guidance_scale'], 
        'num_steps': cfg['evaluation']['num_steps'], 
        'corrector': cfg['evaluation']['corrector'] 
    }
    sampler = samplers.sampling_with_cfg_dpm
    
    return (params, sampler)


def get_ddim(
    cfg: DictConfig,
    model: Module, 
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict, Callable]:
    
    # none is used when param should be passed at runtime
    params = {
        'model': model, 
        't_timesteps': cfg['dataset']['timestep'], 
        'beta': beta, 
        'alpha': alpha, 
        'alpha_bar': alpha_bar, 
        'device': device, 
        'condition': None, 
        'guidance_scale': cfg['evaluation']['guidance_scale'], 
        'num_steps': cfg['evaluation']['num_steps'],  
        'eta': cfg['evaluation']['eta'],  
        'corrector': cfg['evaluation']['corrector'] 
    }
    sampler = samplers.sampling_with_cfg_ddim
    
    return (params, sampler)


sampler_lookup = {
    'DDIM': get_ddim,
    'DPM++': get_dpm,
    'DDPM': get_ddpm
}