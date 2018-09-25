# The language recognizer is a LSTM on character level

import torch
from torch.nn.modules import LSTM

model=LSTM(input_size=256,
           hidden_size=512,
           num_layers=8)
