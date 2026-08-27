import os
import yaml
import torch
import torch.optim as optim
from src.dataset import load_aml_dataset
from src.model import GATv2
from src.utils import FocalLoss, compute_metrics

def main():
    # 1. Load Configurations
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load Dataset (Triggers automatic HF download if data/ is empty)
    print("[1/4] Loading / Downloading dataset into data/ folder...")
    data = load_aml_dataset(root="data")

    # 3. Setup Hardware Device
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[2/4] Using device: {device}")

    # 4. Initialize Model
    model = GATv2(
        in_channels=data.num_node_features,
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=config["model"]["out_channels"],
        edge_dim=data.edge_attr.shape[1] if data.edge_attr is not None else 1,
        heads=config["model"]["heads"],
        dropout=config["model"]["dropout"]
    ).to(device)

    data = data.to(device)
    criterion = FocalLoss(alpha=config["loss"]["alpha"], gamma=config["loss"]["gamma"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # 5. Training Loop
    print("[3/4] Starting training process...")
    model.train()
    for epoch in range(1, config["training"]["epochs"] + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            preds = out.argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(out, dim=-1).detach().cpu().numpy()
            metrics = compute_metrics(data.y.cpu().numpy(), preds, probs)
            print(f"Epoch {epoch:03d}/{config['training']['epochs']} | Loss: {loss.item():.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    # 6. Save Model Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = config["training"]["checkpoint_path"]
    torch.save(model.state_dict(), save_path)
    print(f"[4/4] Model saved successfully to: {save_path}")

if __name__ == "__main__":
    main()