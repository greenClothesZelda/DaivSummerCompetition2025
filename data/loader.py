from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(batch_size:int, num_workers:int, root="./data")->tuple[DataLoader, DataLoader]:
    # Define the transformations for the CIFAR-100 dataset
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        # image의 원래 크기는 32x32이지만 padding을 추가하여 랜덤 크롭을 적용 =>이미지 크기는 동일 그러나 증강됨
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # 랜덤으로 좌우 반전
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean, std)  # 정규화
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean, std)  # 정규화
    ])

    train_set = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        #transform에 위에서 정의한 증강 방식을 적용
        transform=transform_train
    )

    test_set = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        #IOBound를 처리하기 위함
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader