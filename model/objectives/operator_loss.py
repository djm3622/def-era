import torch
import torch.nn as nn

class OperatorLoss(nn.Module):
    def __init__(self, a1, a2):
        super().__init__()
        
        self.l2 = lambda x, y: torch.mean(torch.mean(torch.linalg.norm(x - y, dim=(2, 3), ord=2), dim=1))
        self.l1 = lambda x, y: torch.mean(torch.mean(torch.linalg.norm(x - y, dim=(2, 3), ord=1), dim=1))
        self.a1, self.a2 = a1, a2
        
    def forward(self, y_hat, y):
        err1 = self.l2(y_hat, y) 
        err2 = self.l1(y_hat, y)
        return self.a1 * err1 + self.a2 * err2