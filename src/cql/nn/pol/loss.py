import torch
import torch.nn.functional as F

def mse_loss(predicted: torch.Tensor, 
             target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, target)