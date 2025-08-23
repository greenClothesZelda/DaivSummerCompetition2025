import torch
from torch import nn
from tqdm import tqdm

criteriion = nn.CrossEntropyLoss()

def train_fine_tune(model, train_loader, test_loader, optimizer, scheduler, device, total_epochs):
    model.backbone.eval()
    model.fc.train()
    for epoch in range(total_epochs):
        print(f"Training Pretext Task - Epoch {epoch + 1}/{total_epochs}")
        train_one_epoch(model, train_loader, optimizer, device, epoch=epoch + 1, total_epochs=total_epochs, sheduler=scheduler)
        test(model, test_loader, device)


def train_one_epoch(model, loader, optimizer, device, epoch=None, total_epochs=None, sheduler=None):
    total_loss = 0.0
    desc = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Training"
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in tqdm(loader, desc=desc):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            imgs = model.backbone(imgs)

        optimizer.zero_grad()

        pred = model.fc(imgs)
        loss = criterion(pred, labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    if sheduler:
        sheduler.step()

    avg_loss = total_loss / len(loader)
    tqdm.write(f"Epoch {epoch}/{total_epochs} - Average Loss: {avg_loss:.4f}")

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            imgs = model.backbone(imgs)
            pred = model.fc(imgs)

            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

