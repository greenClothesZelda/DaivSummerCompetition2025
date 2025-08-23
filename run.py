from models.resnet_simclr import ResNetSimCLR
import torch
from torchvision import transforms
from logger import get_logger
from config import *
from data_aug.aug_loader import AugmentedImageDataset
from data_aug.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from train_pretext import train_pretext
from train_fine_tune import train_fine_tune

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
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    ])


    unlabeled_dataset = ImageDataset(root=DATA_PATH, force_download=False, unlabeled=True, transform=resizer)
    unlabeled_loader = AugmentedImageDataset(unlabeled_dataset, transform=aug_transform)

    return DataLoader(unlabeled_loader, batch_size=BATCH_SIZE_PRETEXT, shuffle=True)

def get_train_test_loader():
    IMG_NORM = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resizer = transforms.Compose([
        transforms.Resize(IMG_SIZE),  # Resize Image
        transforms.ToTensor(),  # Convert Image to Tensor
        transforms.Normalize(**IMG_NORM)  # Normalization
    ])

    train_dataset = ImageDataset(root=DATA_PATH, force_download=False, train=True, transform=resizer)
    test_dataset = ImageDataset(root=DATA_PATH, force_download=False, valid=True, transform=resizer)
    return DataLoader(train_dataset, batch_size=BATCH_SIZE_FINE_TUNE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE_FINE_TUNE, shuffle=False)

def main(device):
    unlabeled_loader = get_aug_loader()
    model = ResNetSimCLR(out_dim=NUM_CLASSES)
    model.to(device)
    base_epoch = 0
    try:
        model.backbone.load_state_dict(torch.load(f"models/snapshot/resnet_simclr_epoch_{BASE_EPOCH}.pth"))
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

    logger.info("Pretext training completed and model saved.")

    train_loader, test_loader = get_train_test_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FINE_TUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    train_fine_tune(model=model, train_loader= train_loader, test_loader= test_loader, optimizer=optimizer, device = device, total_epochs= EPOCHS_FINE_TUNE, scheduler=scheduler)
    torch.save(model.state_dict(), "models/snapshot/complete_resnet_simclr_final.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)