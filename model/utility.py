from torch import nn
import torch
import math
import os
from omegaconf import DictConfig, OmegaConf


def load_model_weights(model, state_dict_path):
    pt_load = torch.load(state_dict_path, weights_only=False)
    state_dict = pt_load['model_state_dict']
    
    model_state = model.state_dict()
    matched_weights = {
        k: v for k, v in state_dict.items() 
        if k in model_state and v.shape == model_state[k].shape
    }
    unmatched = set(model_state.keys()) - set(matched_weights.keys())
    if unmatched:
        print(f"Warning - Unmatched keys: {unmatched}")
    
    model.load_state_dict(matched_weights, strict=False)
    
    
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
        
        
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x_encoded = self.encoder(x).pooler_output
                
        x_predicted = self.decoder(x_encoded.view(self.batch_size, 1, self.up_state, self.up_state))
                        
        return x_predicted, y


def load_training_state(accelerator, checkpoint_path, model, optimizer, scheduler):
    state = torch.load(checkpoint_path, map_location=accelerator.device)

    accelerator.unwrap_model(model).load_state_dict(state['model_state_dict'])
    
    optimizer.load_state_dict(state['optimizer_state_dict'])
    
    scheduler.load_state_dict(state['scheduler_state_dict'])
        
        
def save_training_state(accelerator, epoch, model, optimizer, scheduler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }
    
    # Save state
    accelerator.save(state, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    
def get_dict_of_features(
    config: DictConfig
) -> dict:
    
    # Extract pressure levels and features
    pressure_levels = config["features"]["pressure_levels"]
    atmospheric_features = config["features"]["base"]["atmospheric"]
    surface_features = config["features"]["base"]["surface"]

    # Initialize the dictionary and index counter
    features_dict = {}
    index = 0

    # Add atmospheric features with pressure levels
    for feature in atmospheric_features:
        for level in pressure_levels:
            key = f"{feature}_h{level}"
            features_dict[key] = index
            index += 1

    # Add surface features
    for feature in surface_features:
        features_dict[feature] = index
        index += 1
        
    return features_dict


def get_features_swtich(
    config: DictConfig
) -> dict:

    evaluation = config["evaluation"]

    # Extract pressure levels and features
    pressure_levels = evaluation["pressure_levels"]
    atmospheric_features = evaluation["base"]["atmospheric"]
    surface_features = evaluation["base"]["surface"]

    # Initialize the dictionary to store boolean values
    feature_bool_dict = {}

    # Add atmospheric features with pressure levels
    for feature, enabled in atmospheric_features.items():
        for level, level_enabled in pressure_levels.items():
            if level_enabled:  # Only add if the pressure level is enabled
                key = f"{feature}_h{level}"
                feature_bool_dict[key] = enabled

    # Add surface features
    for feature, enabled in surface_features.items():
        feature_bool_dict[feature] = enabled
        
    return feature_bool_dict