import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                 mu: torch.Tensor, logvar: torch.Tensor, 
                 kl_weight: float) -> torch.Tensor:
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss

def visualize_data(original: torch.Tensor, generated: torch.Tensor):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(original[:, 0], original[:, 1], s=5, label='Original')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(generated[:, 0], generated[:, 1], s=5, label='Generated')
    plt.legend()
    plt.show()