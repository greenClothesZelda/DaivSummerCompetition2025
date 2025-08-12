import torch

NUM_WORKERS = 4
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "C:/Users/wlstn/.cache/kagglehub/datasets"