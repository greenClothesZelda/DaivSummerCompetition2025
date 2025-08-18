from data.loader import get_test_loader, get_valid_loader
import matplotlib.pyplot as plt
import numpy as np

def check_test_loader():
    test_loader = get_test_loader()
    x, y = next(iter(test_loader))
    print(f"x shape: {x.shape}, y shape: {y.shape}")

def check_class_images():
    test_loader = get_valid_loader()
    
    # 각 클래스별로 이미지 5개를 수집할 딕셔너리 준비
    class_images = {}
    filled_classes = []  # 5개 이미지가 채워진 클래스들을 순서대로 저장
    
    # test_loader를 순회하면서 각 클래스별로 최대 5개의 이미지 수집
    _max = 0
    for images, labels in test_loader:
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            img = images[i].detach().cpu().numpy()
            label = int(labels[i].item())
            _max = max(_max, label)
            
            if label not in class_images:
                class_images[label] = []
            
            if len(class_images[label]) < 5:
                class_images[label].append(img)
                
                # 5개가 채워졌는지 확인
                if len(class_images[label]) == 5 and label not in filled_classes:
                    filled_classes.append(label)
            
        # 8개 클래스가 모두 채워졌으면 루프 종료
        if len(filled_classes) >= 8:
            break
    print(_max)
    
    # 결과 출력
    print(f"가장 빠르게 채워진 8개 클래스: {filled_classes[:8]}")
    
    # 8개의 클래스에 대해 각각 5개의 이미지 출력
    plt.figure(figsize=(15, 10))
    
    for i, class_id in enumerate(filled_classes[:8]):
        for j in range(5):
            plt.subplot(8, 5, i * 5 + j + 1)
            img = class_images[class_id][j]
            
            # 이미지가 CHW 형식이면 HWC로 변환 (matplotlib 표시용)
            if len(img.shape) == 3 and (img.shape[0] == 1 or img.shape[0] == 3):
                img = np.transpose(img, (1, 2, 0))
                
                # 그레이스케일 이미지인 경우 차원 축소
                if img.shape[2] == 1:
                    img = np.squeeze(img)
            
            # 이미지 값 범위 조정 (0-1 또는 0-255)
            if np.max(img) <= 1.0:
                plt.imshow(img)
            else:
                plt.imshow(img.astype(np.uint8))
                
            plt.title(f"Class {class_id}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #check_test_loader()
    check_class_images()