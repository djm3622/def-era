from torch import nn
import torch
import math


def load_model_weights(model, state_dict):
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