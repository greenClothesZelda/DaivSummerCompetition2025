from torch import nn
import torch.optim as optim

from config import *
from data.loader import get_cifar100_loaders
from models.my_model import SimpleCNN
from train import train_one_epoch
from test import test

from logger import get_logger
logger = get_logger()



def main():
    train_loader, test_loader = get_cifar100_loaders(BATCH_SIZE, NUM_WORKERS)

    model = SimpleCNN(num_classes=100).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test(model, test_loader, criterion, DEVICE)

    # Save the trained model
    torch.save(model.state_dict(), "models/snapshot/simple_cnn.pth")

if __name__ == "__main__":
    main()