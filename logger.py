# logger.py
import logging
import os
import sys

def get_logger(log_file='logs/train.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('cifar100_logger')
    logger.setLevel(logging.INFO)

    # 이미 핸들러 등록된 경우 중복 방지
    if logger.hasHandlers():
        return logger

    # 🔹 파일 핸들러
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 🔹 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger