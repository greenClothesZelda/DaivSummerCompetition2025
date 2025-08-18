from models.resnet_simclr import ResNetSimCLR
import torch
from torchvision import transforms
from logger import get_logger
from config import *
from data_aug.aug_loader import AugmentedImageDataset
from data_aug.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from train_pretext import train_pretext

logger = get_logger()

def get_aug_loader():
    IMG_NORM = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resizer = transforms.Compose([
        transforms.Resize(IMG_SIZE),  # Resize Image
        transforms.ToTensor(),  # Convert Image to Tensor
        transforms.Normalize(**IMG_NORM)  # Normalization
    ])

    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])


    unlabeled_dataset = ImageDataset(root=DATA_PATH, force_download=False, unlabeled=True, transform=resizer)
    unlabeled_loader = AugmentedImageDataset(unlabeled_dataset, transform=aug_transform)

    return DataLoader(unlabeled_loader, batch_size=BATCH_SIZE_PRETEXT, shuffle=True)

def main(device):
    unlabeled_loader = get_aug_loader()
    model = ResNetSimCLR(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM)
    model.to(device)
    base_epoch = 0
    try:
        model.load_state_dict(torch.load(f"models/snapshot/cnn_AE_epoch_{BASE_EPOCH}.pth"))
        logger.info("Pre-trained model loaded successfully.")
        base_epoch = BASE_EPOCH
    except FileNotFoundError:
        logger.info("No pre-trained model found, starting from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(unlabeled_loader),
                                                           eta_min=0, last_epoch=-1)
    train_pretext(model=model.backbone, loader=unlabeled_loader, optimizer=optimizer, scheduler=scheduler,
                  device=device, base_epoch=base_epoch, total_epochs=EPOCHS_PRETEXT)

    torch.save(model.state_dict(), "models/snapshot/resnet_simclr_final.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)