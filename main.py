from torch import nn
import torch.optim as optim

from config import *
from data.loader import get_unlabeled_loader
from models.cnn_autoencoder import CNNAutoencoder
from train import train_one_epoch

from logger import get_logger
logger = get_logger()

BASE_EPOCH = 50

def main():
    train_loader = get_unlabeled_loader(batch_size=32, num_workers=2)

    model = CNNAutoencoder().to(DEVICE)
    base_epoch = 0
    try:
        model.load_state_dict(torch.load(f"models/snapshot/cnn_AE_epoch_{BASE_EPOCH}.pth"))
        logger.info("Pre-trained model loaded successfully.")
        base_epoch = BASE_EPOCH
    except FileNotFoundError:
        logger.info("No pre-trained model found, starting from scratch.")


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1 + base_epoch}/{EPOCHS+base_epoch}")
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch=epoch + 1, total_epochs=EPOCHS)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Saving model at epoch {epoch + 1}")
            torch.save(model.state_dict(), f"models/snapshot/cnn_AE_epoch_{epoch + 1+base_epoch}.pth")

    # Save the trained autoencoder
    torch.save(model.state_dict(), "models/snapshot/cnn_AE_final.pth")

if __name__ == "__main__":
    main()
