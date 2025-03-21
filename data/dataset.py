import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def generate_data(n_samples:int, noise:float, random_seed:int)-> torch.Tensor:
    np.random.seed(random_seed)
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    scaler=StandardScaler()
    X = scaler.fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), scaler

def inverse_transform(data:torch.Tensor, scaler:StandardScaler)->np.ndarray:
    return scaler.inverse_transform(data.numpy())