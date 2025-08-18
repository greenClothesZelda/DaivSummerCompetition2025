import torch
from torch import nn

class CNNDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=3):
        super().__init__()
        
        # 4x4 크기의 특징 맵에서 시작하도록 디자인
        self.decoder = nn.Sequential(
            # 입력 특징을 4x4 공간 해상도로 변환
            nn.Linear(in_channels, 4 * 4 * 256),
            nn.ReLU(inplace=True),
            
            # 특징 맵 재구성 (reshape)
            nn.Unflatten(1, (256, 4, 4)),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 최종 출력층 (64x64x3)
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 출력값을 [-1, 1] 범위로 제한 (선택적)
        )
        
    def forward(self, x):
        # 입력이 벡터가 아니라 특징 맵인 경우를 위한 분기 처리
        if len(x.shape) > 2:
            return self.decoder_from_features(x)
        return self.decoder(x)
    
    def decoder_from_features(self, x):
        # 이미 특징 맵 형태인 경우, Linear와 Unflatten 단계를 건너뛰고 적절한 레이어부터 시작
        # 입력 크기에 따라 적절한 레이어 선택
        shape = x.shape
        
        if shape[2] == 4 and shape[3] == 4:
            # 4x4 특징 맵인 경우
            for i, layer in enumerate(self.decoder):
                if i >= 3:  # Linear와 ReLU, Unflatten 건너뛰기
                    x = layer(x)
        elif shape[2] == 8 and shape[3] == 8:
            # 8x8 특징 맵인 경우
            for i, layer in enumerate(self.decoder):
                if i >= 7:  # 첫 번째 ConvTranspose2d 이후부터 시작
                    x = layer(x)
        elif shape[2] == 16 and shape[3] == 16:
            # 16x16 특징 맵인 경우
            for i, layer in enumerate(self.decoder):
                if i >= 11:  # 두 번째 ConvTranspose2d 이후부터 시작
                    x = layer(x)
        elif shape[2] == 32 and shape[3] == 32:
            # 32x32 특징 맵인 경우
            for i, layer in enumerate(self.decoder):
                if i >= 15:  # 세 번째 ConvTranspose2d 이후부터 시작
                    x = layer(x)
                    
        return x

if __name__ == "__main__":
    decoder = CNNDecoder()

    # 테스트용 입력 벡터
    x = torch.randn(1, 128)  # Batch size of 1, latent vector of size 128
    reconstructed_image = decoder(x)
    print(reconstructed_image.shape)  # Should print the shape of the reconstructed image (1, 3, 64, 64)
