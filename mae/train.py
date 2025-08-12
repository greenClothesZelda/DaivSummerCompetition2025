from tqdm import tqdm
from logger import get_logger
logger = get_logger()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0

    desc = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Training"
    for batch in tqdm(loader, desc=desc):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        optimizer.zero_grad()
        reconstructed, _ = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    logger.info(f"Train Reconstruction Loss: {avg_loss:.4f}")