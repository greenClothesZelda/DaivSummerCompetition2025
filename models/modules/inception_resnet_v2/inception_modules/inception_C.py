import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionC(nn.Module):
    def __init__(self):
        super().__init__()