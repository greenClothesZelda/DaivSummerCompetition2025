import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from models.masked_autoencoder import get_mae_model
from data.loader import get_unlabeled_loader
from models.cnn_autoencoder import CNNAutoencoder

def show_image_comparison(original_imgs, reconstructed_imgs, num_images=10):
    plt.figure(figsize=(num_images * 2, 4))
    for i in range(num_images):
        # 원본 이미지
        plt.subplot(2, num_images, i + 1)
        img = original_imgs[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        if i == 0:
            plt.ylabel('Original')

        # 복원 이미지
        plt.subplot(2, num_images, num_images + i + 1)
        rec_img = reconstructed_imgs[i].permute(1, 2, 0).cpu().numpy()
        rec_img = np.clip(rec_img, 0, 1)
        plt.imshow(rec_img)
        plt.axis('off')
        if i == 0:
            plt.ylabel('Reconstructed')
    plt.tight_layout()
    plt.show()

def reconstruct_images(model, imgs, device):
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        reconstructed_imgs, _ = model(imgs)
    return reconstructed_imgs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = get_unlabeled_loader()
    imgs, _ = next(iter(loader))
    imgs = imgs[:10]

    # 모델 로드
    model = CNNAutoencoder().to(device)
    checkpoint = torch.load('../models/snapshot/cnn_AE_final.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # 복원 이미지 얻기
    reconstructed_imgs = reconstruct_images(model, imgs, device)

    # 비교 시각화
    show_image_comparison(imgs, reconstructed_imgs, num_images=10)

if __name__ == '__main__':
    main()

