# The language recognizer is a LSTM on character level

import torch
from torch.nn.modules import LSTM
import torch.nn as nn


class LSTMWrapper(nn.Module):
    def __init__(self, output_size=4, hidden_size=256, *args, **kwargs):
        super(LSTMWrapper, self).__init__()
        self.lstm = LSTM(hidden_size=hidden_size,*args, **kwargs)
        self.output = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, input, hx=None):
        output, statetuple = self.lstm(input, hx)
        output=output[:,-1,:]
        output=self.output(output)
        # this is a design decision that can be experimented with
        return output

def build_model():
    # model will predict at each input character.
    computer = LSTMWrapper(input_size=256,
                           hidden_size=512,
                           num_layers=8)
    return computer