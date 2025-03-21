import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        activation: str = "leaky_relu",
        batchnorm: bool = True
    ):
        super().__init__()
        
        # 编码器
        self.encoder = self._build_mlp(
            [input_dim] + encoder_dims,
            activation,
            batchnorm
        )
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # 解码器
        self.decoder = self._build_mlp(
            [latent_dim] + decoder_dims + [input_dim],
            activation,
            batchnorm
        )

    def _build_mlp(self, dims: List[int], activation: str, batchnorm: bool) -> nn.Sequential:
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif activation == "relu":
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    ##############################
    # 测试代码 (可直接运行验证结构) #
    ##############################
if __name__ == "__main__":
    # 快速测试模型结构
    model = VAE(
        input_dim=2,
        latent_dim=5,
        encoder_dims=[32, 64, 128],
        decoder_dims=[128, 64, 32],
        batchnorm=True
    )
    x = torch.randn(10, 2)  # 批量大小=10
    recon_x, mu, logvar = model(x)
    assert recon_x.shape == x.shape, "输入输出维度不匹配!"
    print("模型测试通过!")