from models.modules.cnn_autoencoder.cnn_encoder import CNNEncoder
from config import *
from data.loader import get_train_loader, get_test_loader, get_test_dataset, get_train_dataset, get_valid_loader, get_valid_dataset
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm  # tqdm 임포트 추가
from pathlib import Path

def get_model():
    model = CNNEncoder().to(DEVICE)
    try:
        print("Loading pre-trained model...")
        state_dict = torch.load('../models/snapshot/cnn_AE_final.pth', map_location='cpu')
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        model.load_state_dict(encoder_state_dict, strict=True)
    except FileNotFoundError:
        print("No pre-trained model found")

    model.eval()
    return model


def create_data_table(model):
    test_loader = get_train_loader(batch_size=64, num_workers=1)
    train_dataset = get_train_dataset()  # 클래스 이름을 가져오기 위한 데이터셋
    vector_list = {}

    # 벡터 저장 디렉토리 확인 및 생성
    Path("vectors").mkdir(exist_ok=True)

    for data in tqdm(test_loader, desc="잠재 벡터 추출 중"):
        imgs, classes = data
        latent_vectors = model(imgs.to(DEVICE))

        for i, latent_vector in enumerate(latent_vectors):
            class_index = classes[i].item()
            class_name = train_dataset.classes[class_index]  # 클래스 인덱스를 실제 이름으로 변환

            if class_name not in vector_list:
                vector_list[class_name] = []
            vector_list[class_name].append(latent_vector.cpu().detach().numpy())
    total = 0
    for class_name in tqdm(vector_list, desc="벡터 저장 중"):
        vector_list[class_name] = np.stack(vector_list[class_name], axis=0)
        np.save(f"vectors/{class_name}.npy", vector_list[class_name])
        total += vector_list[class_name].shape[0]

    print("잠재 벡터가 추출되어 클래스 이름으로 저장되었습니다., 총 벡터 수:", total)

def knn_classify(k=5, model=None):
    test_dataset = get_test_dataset()
    train_dataset = get_train_dataset()

    # 클래스 이름으로 저장된 벡터 파일 로드
    vector_list = {}
    for class_index, class_name in enumerate(train_dataset.classes):
        try:
            vector_list[class_name] = np.load(f"vectors/{class_name}.npy")
        except FileNotFoundError:
            print(f"경고: {class_name} 클래스의 벡터 파일을 찾을 수 없습니다.")
            continue

    # 학습 벡터 리스트 생성
    train_vectors = []
    for class_name, vectors in vector_list.items():
        for vector in vectors:
            train_vectors.append((vector, class_name))

    loader = get_test_loader(batch_size=50, num_workers=1)
    total = 0
    results = []

    for data in tqdm(loader, desc="KNN 분류 진행중"):
        imgs, ids = data
        latent_vectors = model(imgs.to(DEVICE))
        latent_vectors = latent_vectors.cpu().detach().numpy()

        for i, latent_vector in enumerate(latent_vectors):
            # 현재 테스트 벡터와 모든 학습 벡터 사이의 거리 계산
            distances = []
            for train_vector, train_label in train_vectors:
                dist = np.linalg.norm(latent_vector - train_vector)
                distances.append((dist, train_label))

            # 거리를 기준으로 정렬하여 가장 가까운 이웃 찾기
            distances.sort(key=lambda x: x[0])

            # 상위 k개 이웃의 클래스 가져오기
            top_k_classes = [label for dist, label in distances[:k]]

            # 다수결 투표로 클래스 예측
            predicted_class = Counter(top_k_classes).most_common(1)[0][0]

            total += 1
            pred_id = test_dataset.classes[ids[i].item()]
            results.append((pred_id, predicted_class))  # 클래스 이름 직접 사용

    print(f"총 {total}개의 샘플에 대한 예측 완료.")

    import csv
    output_file = "test.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

    print(f"예측 결과가 {output_file}에 저장되었습니다.")

def compute_class_stats(model=None):
    train_dataset = get_train_dataset()
    class_means = {}
    class_vars = {}
    all_vectors = []

    # 저장된 벡터 파일 불러오기
    for class_name in tqdm(train_dataset.classes, desc="Computing class stats"):
        file_path = f"vectors/{class_name}.npy"
        try:
            vectors = np.load(file_path)
            class_mean = np.mean(vectors, axis=0)
            class_var = np.var(vectors, axis=0)
            class_means[class_name] = class_mean
            class_vars[class_name] = class_var
            all_vectors.append(vectors)

            print(f"Class {class_name}:", end=",")
            # print(f"  Mean: {class_mean}", end=", ")
            print(f"  Variance: {np.mean(class_var)}")
        except FileNotFoundError:
            print(f"Warning: Could not find vectors for class {class_name}")
            continue

    # 전체 평균과 분산 계산
    if not all_vectors:
        print("No vectors found to compute overall stats.")
        return

    all_vectors = np.concatenate(all_vectors, axis=0)
    total_var = np.var(all_vectors, axis=0)

    print("\nOverall:")
    print(f"  Variance: {np.mean(total_var)}")


def validate(model=None, k=5):
    train_dataset = get_train_dataset()
    loader = get_valid_loader(batch_size=50, num_workers=1)
    dataset = get_valid_dataset()

    # 클래스 이름으로 저장된 벡터 파일 로드
    vector_list = {}
    for class_index, class_name in enumerate(train_dataset.classes):
        try:
            vector_list[class_name] = np.load(f"vectors/{class_name}.npy")
        except FileNotFoundError:
            print(f"경고: {class_name} 클래스의 벡터 파일을 찾을 수 없습니다.")
            continue

    # 학습 벡터 리스트 생성
    train_vectors = []
    for class_name, vectors in vector_list.items():
        for vector in vectors:
            train_vectors.append((vector, class_name))

    total = 0
    correct = 0

    for data in tqdm(loader, desc="Validating"):
        imgs, labels = data
        latent_vectors = model(imgs.to(DEVICE))
        latent_vectors = latent_vectors.cpu().detach().numpy()

        for i, latent_vector in enumerate(latent_vectors):
            distances = []
            for train_vector, train_label in train_vectors:
                dist = np.linalg.norm(latent_vector - train_vector)
                distances.append((dist, train_label))

            distances.sort(key=lambda x: x[0])
            top_k_classes = [label for dist, label in distances[:k]]
            predicted_class = Counter(top_k_classes).most_common(1)[0][0]

            true_label_index = labels[i].item()
            true_label_name = dataset.classes[true_label_index]

            if predicted_class == true_label_name:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    with torch.no_grad():
        model = get_model()
        create_data_table(model)
        validate(model=model, k=5)
        knn_classify(k=5, model=model)
        compute_class_stats(model=model)


