# The language recognizer is a LSTM on character level

import torch
from torch.nn.modules import LSTM
import torch.nn as nn


class LSTMWrapper(nn.Module):
    def __init__(self, output_size=12, hidden_size=256, *args, **kwargs):
        super(LSTMWrapper, self).__init__()
        self.lstm = LSTM(hidden_size=hidden_size,*args, **kwargs)
        self.output = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, input, hx=None):
        output, statetuple = self.lstm(input, hx)
        # this is a design decision that can be experimented with
        output=self.output(output)
        # output=torch.max(output,dim=1)[0]
        output=output[:,-1,:]
        return output

def build_model():
    # model will predict at each input character.
    computer = LSTMWrapper(input_size=256,
                           hidden_size=512,
                           num_layers=32)
    return computer


class BOW_model(nn.Module):
    def __init__(self, output_size=12, hidden_factor=64):
        super(BOW_model, self).__init__()
        self.pipeline=nn.Sequential(
            nn.Linear(10000,hidden_factor*8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_factor*8,hidden_factor*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_factor*2, output_size)
        )
        self.reset_parameters()

    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        self.apply(self.reset_mod)

    def forward(self, input):
        output=self.pipeline(input)
        return output
    

class LSTM_vocab(nn.Module):
    def __init__(self, vocab_size=50000, vocab_embed_d=512, output_size=12, hidden_size=256, *args, **kwargs):
        super(LSTM_vocab, self).__init__()
        self.src_word_emb = nn.Embedding(
            vocab_size, vocab_embed_d, padding_idx=0)
        self.lstm = LSTM(input_size=vocab_embed_d, hidden_size=hidden_size,*args, **kwargs)
        self.output = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, input, hx=None):
        input=self.src_word_emb(input)
        output, statetuple = self.lstm(input, hx)
        # this is a design decision that can be experimented with
        output=self.output(output)
        # output=torch.max(output,dim=1)[0]
        output=output[:,-1,:]
        return output


class BOW_vocab(nn.Module):
    def __init__(self, vocab_size=50000, output_size=12, hidden_factor=64):
        super(BOW_vocab, self).__init__()
        self.l1=nn.Linear(vocab_size, hidden_factor * 16)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout()
        self.l2=nn.Linear(hidden_factor*16, hidden_factor * 8)
        self.l3=nn.Linear(hidden_factor * 8, hidden_factor * 2)
        self.l4=nn.Linear(hidden_factor * 2, output_size)
        self.reset_parameters()

    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        self.apply(self.reset_mod)

    def forward(self, input):
        o = self.l1(input)
        o = self.relu(o)
        o = self.dropout(o)
        o = self.l2(o)
        o = self.relu(o)
        o = self.dropout(o)
        o = self.l3(o)
        o = self.relu(o)
        o = self.dropout(o)
        output = self.l4(o)
        return output
