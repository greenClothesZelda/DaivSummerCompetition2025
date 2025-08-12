from data.loader import get_unlabeled_loader
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow_grid(imgs, nrow=5):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)

    npimg = grid.numpy().transpose((1, 2, 0))
    npimg = std * npimg + mean  # 정규화 해제
    npimg = np.clip(npimg, 0, 1)
    plt.figure(figsize=(8,8))
    plt.imshow(npimg)
    plt.axis('off')
    plt.show()

def main():
    test_loader = get_unlabeled_loader()
    x, y = next(iter(test_loader))
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    imshow_grid(x, nrow=5)

if __name__ == "__main__":
    main()