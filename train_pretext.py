import torch
from torch import nn
from tqdm import tqdm
from config import *
import torch.nn.functional as F

criteriion = nn.CrossEntropyLoss()

def train_pretext(model, loader, optimizer, scheduler, device, base_epoch, total_epochs):
    model.train()
    total_epochs = base_epoch + total_epochs
    for epoch in range(base_epoch, total_epochs):
        print(f"Training Pretext Task - Epoch {epoch + 1}/{total_epochs}")
        train_one_epoch(model, loader, optimizer, scheduler, device, epoch=epoch + 1, total_epochs=total_epochs)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/snapshot/resnet_simclr_epoch_{epoch + 1}.pth")


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch=None, total_epochs=None, temperature=0.07):
    total_loss = 0.0
    desc = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Training"
    criterion = nn.CrossEntropyLoss()

    for img_ori, img_aug in tqdm(loader, desc=desc):
        batch_size = img_ori.shape[0]

        img_ori = img_ori.to(device)
        img_aug = img_aug.to(device)

        optimizer.zero_grad()

        images = torch.cat([img_ori, img_aug], dim=0)

        # latents.shape = (2 * batch_size, latent_dim)
        latents = model(images)
        latents = F.normalize(latents, dim=1)

        # 코사인 유사도 행렬 계산
        similarity_matrix = torch.matmul(latents, latents.T) / temperature

        # 올바른 마스킹 생성
        # 각 샘플의 positive pair는 원본과 증강 이미지 쌍임
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=bool).to(device)

        # 첫 번째 batch_size 행에서는 batch_size 이후의 같은 인덱스가 positive pair
        # 두 번째 batch_size 행에서는 처음 batch_size 내의 같은 인덱스가 positive pair
        for i in range(batch_size):
            mask[i, batch_size + i] = True
            mask[batch_size + i, i] = True

        # 자기 자신은 마스킹
        mask_self = torch.eye(2 * batch_size, dtype=bool).to(device)

        # 자기 자신은 제외하고, positive pair가 아닌 모든 요소를 negative로 사용
        negative_mask = ~(mask | mask_self)

        # positive pair에 대한 로짓 추출
        positive_logits = torch.masked_select(similarity_matrix, mask).view(2 * batch_size, 1)
        # negative pair에 대한 로짓 추출
        negative_logits = similarity_matrix.masked_fill(~negative_mask, -float('inf'))

        # 모든 로짓을 결합 (각 샘플의 positive pair와 모든 negative pairs)
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        # 첫 번째 열이 positive pair이므로 labels는 모두 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(device)

        loss = criterion(logits, labels)

        # 역전파
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    tqdm.write(f"Epoch {epoch}/{total_epochs} - Average Loss: {avg_loss:.4f}")