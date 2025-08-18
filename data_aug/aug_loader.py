import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

class AugmentedImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img_origin, id = self.dataset[index]

        return img_origin, self.transform(img_origin)

    def __len__(self):
        return len(self.dataset)

def main():
    # Image Resizing and Tensor Conversion
    IMG_SIZE = (64, 64)

    # ImageNet Normalization
    IMG_NORM = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resizer = transforms.Compose([
        transforms.Resize(IMG_SIZE),  # Resize Image
        transforms.ToTensor(),  # Convert Image to Tensor
        transforms.Normalize(**IMG_NORM)  # Normalization
    ])

    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])

    from data_aug import image_dataset
    from matplotlib import pyplot as plt
    test_dataset = image_dataset.ImageDataset(root='../data', force_download=False, unlabeled=True, transform=resizer)
    dataset = AugmentedImageDataset(test_dataset, transform=simclr_transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    for img_origin, img_aug in loader:
        print(img_origin.shape, img_aug.shape)
        # Visualize all batch of the original and augmented images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(img_origin[i].permute(1, 2, 0).numpy())
            axes[0, i].axis('off')
            axes[1, i].imshow(img_aug[i].permute(1, 2, 0).numpy())
            axes[1, i].axis('off')
        plt.show()

        break  # Just to test one batch
if __name__ == "__main__":
    main()