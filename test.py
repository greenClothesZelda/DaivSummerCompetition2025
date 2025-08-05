from logger import get_logger
logger = get_logger()

import torch
def test(model, loader, criterion, device):
    # 모델을 평가 모드로 변경
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            running_loss += loss.item()
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    logger.info(f"Test Loss: {running_loss / len(loader):.4f}, Acc: {acc:.2f}%")