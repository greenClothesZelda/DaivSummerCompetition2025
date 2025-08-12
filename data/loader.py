from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import config

from data.competition_data import ImageDataset

# Image Resizing and Tensor Conversion
IMG_SIZE = (64, 64)

# ImageNet Normalization
IMG_NORM = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

resizer = transforms.Compose([
    transforms.Resize(IMG_SIZE), # Resize Image
    transforms.ToTensor(), # Convert Image to Tensor
    transforms.Normalize(**IMG_NORM) # Normalization
])

valid_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(**IMG_NORM)
])

path = Path(config.DATA_PATH)

print(f"Dataset Path: {path}")
print(path.is_absolute())


train_dataset = ImageDataset(root=path, force_download=False, train=True, transform=resizer)
valid_dataset = ImageDataset(root=path, force_download=False, valid=True, transform=valid_transform)
test_dataset = ImageDataset(root=path, force_download=False, train=False, transform=resizer)
unlabeled_dataset = ImageDataset(root=path, force_download=False, unlabeled=True, transform=resizer)


if __name__ == "__main__":
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Valid Dataset Size: {len(valid_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    print(f"Unlabeled Dataset Size: {len(unlabeled_dataset)}")