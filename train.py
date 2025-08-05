from logger import get_logger
logger = get_logger()

def train_one_epoch(model, loader, criterion, optimizer, device):
    #모델을 학습모드로 변경
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct +=(preds.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    acc = 100. * correct / total
    logger.info(f"Train Loss: {running_loss / len(loader):.4f}, Acc: {acc:.2f}%")