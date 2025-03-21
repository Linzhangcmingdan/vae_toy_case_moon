import torch
import yaml
from torch import optim
from model import VAE
from data.dataset import generate_data, inverse_transform
from tool.tools import loss_function, visualize_data

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置
    config = load_config("config.yaml")
    
    # 生成数据
    X, scaler = generate_data(
        n_samples=config["data"]["n_samples"],
        noise=config["data"]["noise"],
        random_seed=config["data"]["random_seed"]
    )
    
    # 初始化模型
    model = VAE(
        input_dim=config["data"]["input_dim"],
        latent_dim=config["data"]["latent_dim"],
        encoder_dims=config["model"]["encoder_dims"],
        decoder_dims=config["model"]["decoder_dims"],
        activation=config["model"]["activation"],
        batchnorm=config["model"]["use_batchnorm"]
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"]
    )
    
    # 训练循环
    for epoch in range(config["training"]["epochs"]):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(X)
        loss = loss_function(
            recon_x, X, mu, logvar,
            kl_weight=config["training"]["kl_weight"]
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 生成并可视化数据
    with torch.no_grad():
        z = torch.randn(1100, config["data"]["latent_dim"])
        generated = model.decode(z)
        generated = inverse_transform(generated, scaler)
    
    visualize_data(X.numpy(), generated)

if __name__ == "__main__":
    main()