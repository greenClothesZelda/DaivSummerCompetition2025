import torch

NUM_WORKERS = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "C:/Users/wlstn/.cache/kagglehub/datasets"