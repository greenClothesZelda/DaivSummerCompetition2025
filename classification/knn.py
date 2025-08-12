from models.modules.cnn_autoencoder.cnn_encoder import CNNEncoder
from config import *
from data.loader import get_train_loader, get_test_loader
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm  # tqdm 임포트 추가

model = CNNEncoder().to(DEVICE)
try:
    print("Loading pre-trained model...")
    state_dict = torch.load('../models/snapshot/cnn_AE_final.pth', map_location='cpu')
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    model.load_state_dict(encoder_state_dict, strict=True)
except FileNotFoundError:
    print("No pre-trained model found")

model.eval()

def create_data_table():
    test_loader = get_train_loader(batch_size=64, num_workers=1)
    vector_list = {}
    for data in test_loader:
        imgs, classes = data
        latent_vectors = model(imgs.to(DEVICE))

        for i, latent_vector in enumerate(latent_vectors):
            class_name = classes[i].item()
            if class_name not in vector_list:
                vector_list[class_name] = []
            vector_list[class_name].append(latent_vector.cpu().detach().numpy())

    for key in vector_list:
        vector_list[key] = np.stack(vector_list[key], axis=0)
        np.save("vectors/" + str(key) + ".npy", vector_list[key])
    print("Latent vectors extracted and stored in vector_list.")

def knn_classify(k=5):
    #load the vectors
    vector_list = {}
    for i in range(200):
        vector_list[i] = np.load("vectors/" + str(i) + ".npy")

    # Flatten the vector_list into a list of (vector, class_label) tuples
    # This is the training set for KNN
    train_vectors = []
    for class_label, vectors in vector_list.items():
        for vector in vectors:
            train_vectors.append((vector, class_label))

    loader = get_train_loader(batch_size=64, num_workers=1)
    correct = 0
    total = 0

    results = []

    for data in tqdm(loader, desc="KNN 분류 진행중"):  # tqdm으로 감싸기
        imgs, ids = data
        # These are the test vectors
        latent_vectors = model(imgs.to(DEVICE))
        latent_vectors = latent_vectors.cpu().detach().numpy()

        for i, latent_vector in enumerate(latent_vectors):
            # Calculate distance from the current test vector to all training vectors
            distances = []
            for train_vector, train_label in train_vectors:
                dist = np.linalg.norm(latent_vector - train_vector)
                distances.append((dist, train_label))

            # Sort distances to find the nearest neighbors
            distances.sort(key=lambda x: x[0])

            # Get the classes of the top k neighbors
            top_k_classes = [label for dist, label in distances[:k]]

            # Predict the class by majority vote
            predicted_class = Counter(top_k_classes).most_common(1)[0][0]

            # if predicted_class == labels[i].item():
            #     correct += 1
            total += 1
            results.append((ids[i].item(), predicted_class))
    #print(f"Accuracy: {correct / total * 100:.2f}%")

    import csv

    output_file = "submissions.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])  # 헤더 작성
        writer.writerows(results)  # 데이터 작성

    print(f"예측 결과가 {output_file}에 저장되었습니다.")

def compute_class_stats():
    class_means = {}
    class_vars = {}
    all_vectors = []

    # 저장된 벡터 파일 불러오기
    for class_idx in range(200):
        file_path = f"vectors/{class_idx}.npy"
        vectors = np.load(file_path)
        class_mean = np.mean(vectors, axis=0)
        class_var = np.var(vectors, axis=0)
        class_means[class_idx] = class_mean
        class_vars[class_idx] = class_var
        all_vectors.append(vectors)

        print(f"Class {class_idx}:", end=",")
        #print(f"  Mean: {class_mean}", end=", ")
        print(f"  Variance: {np.mean(class_var)}")

    # 전체 평균과 분산 계산
    all_vectors = np.concatenate(all_vectors, axis=0)
    total_mean = np.mean(all_vectors, axis=0)
    total_var = np.var(all_vectors, axis=0)

    print("\nOverall:")
    print(f"  Variance: {np.mean(total_var)}")

if __name__ == "__main__":
    with torch.no_grad():
        #create_data_table()
        knn_classify(k=5)
        # compute_class_stats()

