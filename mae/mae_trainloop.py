from data.loader import get_unlabeled_loader
from models.masked_autoencoder import get_mae_model
from mae.train import train

import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mae_model().to(device)
    # Load the model state if available
    try:
        model.load_state_dict(torch.load("../models/snapshot/mae.pth"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found, starting from scratch.")

    unlabeled_loader = get_unlabeled_loader(batch_size=64, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    train(
        model=model,
        dataloader=unlabeled_loader,
        optimizer=optimizer,
        device=device,
        epochs=50
    )

    torch.save(model.state_dict(), "../models/snapshot/mae.pth")

if __name__ == "__main__":
    main()
