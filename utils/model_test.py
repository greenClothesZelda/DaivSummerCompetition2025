import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from config import DEVICE
from data.loader import get_cifar100_loaders
from models.my_model import SimpleCNN
from torchvision.datasets import CIFAR100

def imshow_grid_with_labels(imgs, origins, preds, class_names, nrow=5):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    imgs = imgs.cpu()
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)
    npimg = grid.numpy().transpose((1, 2, 0))
    npimg = std * npimg + mean
    npimg = np.clip(npimg, 0, 1)

    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    for i in range(nrow * nrow):
        row, col = divmod(i, nrow)
        ax = axes[row, col]
        img = imgs[i].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        origin = class_names[origins[i]]
        pred = class_names[preds[i]]
        color = 'blue' if origins[i] == preds[i] else 'red'
        ax.set_title(f"origin: {origin}\npred: {pred}", color=color, fontsize=8)
    plt.tight_layout()
    plt.show()

def main():
    # 모델 로드
    model = SimpleCNN(num_classes=100)
    model.load_state_dict(torch.load("../models/snapshot/simple_cnn.pth"))
    model.eval()
    model.to(DEVICE)

    # 데이터 로더 및 클래스 이름
    _, test_loader = get_cifar100_loaders(batch_size=25, num_workers=0, root="../data")
    class_names = CIFAR100(root="../data", train=False, download=True).classes

    # 배치 하나 추출
    x, y = next(iter(test_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        preds = outputs.argmax(dim=1).cpu().numpy()
    y = y.cpu().numpy()

    imshow_grid_with_labels(x, y, preds, class_names, nrow=5)

if __name__ == "__main__":
    main()