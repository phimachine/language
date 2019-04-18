from fileinput import *
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from model import *
from torch.optim import Adam
from torch.autograd.variable import Variable
from os.path import abspath
from pathlib import Path
import torch
from collections import deque
import numpy as np
from tqdm import tqdm

def train(training_data, validation_data, model, optimizer, starting_epoch, num_epochs, id_str, n_batches, input_long=False):
    dq=deque(maxlen=100)

    for epoch in range(starting_epoch, num_epochs):
        for i,data_point in tqdm(enumerate(training_data), total=n_batches):
            model.train()
            optimizer.zero_grad()
            input, target= data_point
            if input_long:
                input=input.long()
            else:
                input=input.float()
            target=target.long()
            target=target.squeeze(1)

            input=Variable(input)
            target=Variable(target)
            input=input.cuda()
            target=target.cuda()
            y=model(input)
            loss=cross_entropy(y,target)
            loss.backward()
            optimizer.step()

            dq.appendleft(float(loss.item()))
            # if i % 100==0:
            #     print(float(loss.item()))

            if i % 100==0:
                print("\nepoch", epoch, "iteration", i,"loss",float(loss.item()),"running", np.mean(dq))

            if i % 10000==0:
                val_losses = []

                save_model(model, optimizer, epoch, i, id_str)
                for j, data_point in enumerate(validation_data):
                    if j<100:
                        model.eval()
                        input, target = data_point
                        if input_long:
                            input=input.long()
                        else:
                            input = input.float()
                        target = target.long()
                        target = target.squeeze(1)
                        input = Variable(input)
                        target = Variable(target)
                        input = input.cuda()
                        target = target.cuda()
                        y = model(input)
                        loss=cross_entropy(y,target)
                        val_losses.append(float(loss.item()))
                    else:
                        break
                print("\nepoch", epoch, "validation loss", np.mean(val_losses))


def save_model(net, optim, epoch, iteration, savestr):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    if not os.path.isdir(Path(task_dir) / "saves" / savestr):
        os.mkdir(Path(task_dir) / "saves" / savestr)
    pickle_file = Path(task_dir).joinpath("saves/" + savestr + "/lan_" + str(epoch) + "_" + str(iteration) + ".pkl")
    with pickle_file.open('wb') as fhand:
        torch.save((net, optim, epoch, iteration), fhand)
    print('model saved at', pickle_file)



def load_model(computer, optim, starting_epoch, starting_iteration, savestr):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves" / savestr
    highestepoch = 0
    highestiter = 0
    for child in save_dir.iterdir():
        try:
            epoch = str(child).split("_")[1]
            iteration = str(child).split("_")[2].split('.')[0]
        except IndexError:
            print(str(child))
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if child.stat().st_size > 20480:
            if epoch > highestepoch or (iteration > highestiter and epoch == highestepoch):
                highestepoch = epoch
                highestiter = iteration
    if highestepoch == 0 and highestiter == 0:
        print("nothing to load")
        return computer, optim, starting_epoch, starting_iteration
    pickle_file = Path(task_dir).joinpath(
        "saves/" + savestr + "/lan_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at", pickle_file)
    with pickle_file.open('rb') as pickle_file:
        computer, optim, epoch, iteration = torch.load(pickle_file)
    print('Loaded model at epoch ', highestepoch, 'iteartion', iteration)

    return computer, optim, highestepoch, highestiter


def main_train():
    computer=build_model()
    computer.cuda()
    computer.reset_parameters()
    lan_dic={"C":0,
             "C#":1,
             "C++":2,
             "Go":3,
             "Java":4,
             "Javascript":5,
             "Lua":6,
             "Objective-C":7,
             "Python":8,
             "Ruby":9,
             "Rust":10,
             "Shell":11}
    t, v= get_data_set(lan_dic)
    optimizer=Adam(computer.parameters(),lr=0.001)
    ig=TimeSeriesIG(t, lan_dic, 512)
    dl=DataLoader(ig,batch_size=16,shuffle=True,num_workers=0)
    train(dl, computer, optimizer)

def bigram_train():
    computer= LSTMWrapper(input_size=10000,
                           hidden_size=128,
                           num_layers=8)
    computer.cuda()
    computer.reset_parameters()
    lan_dic={"C":0,
             "C#":1,
             "C++":2,
             "Go":3,
             "Java":4,
             "Javascript":5,
             "Lua":6,
             "Objective-C":7,
             "Python":8,
             "Ruby":9,
             "Rust":10,
             "Shell":11}
    t, v= get_data_set(lan_dic)
    optimizer=Adam(computer.parameters(),lr=0.001)
    ig=TimeSeriesIG(t, lan_dic, 512, bigram=True)
    dl=DataLoader(ig,batch_size=32,shuffle=True,num_workers=0)
    train(dl, computer, optimizer)



def bigram_bow_train(load=False, resplit=False):
    computer= BOW_model(hidden_factor=64)
    computer.cuda()
    computer.reset_parameters()

    lan_dic={"C":0,
             "C#":1,
             "C++":2,
             "Go":3,
             "Java":4,
             "Javascript":5,
             "Lua":6,
             "Objective-C":7,
             "Python":8,
             "Ruby":9,
             "Rust":10,
             "Shell":11}
    t, v= get_data_set(lan_dic, load=not resplit, save=resplit)
    optimizer=Adam(computer.parameters(),lr=0.001)

    if load:
        computer, optimizer, highestepoch, highestiter = load_model(computer, optimizer, 0, 0, "full")

    tig=TimeSeriesIG(t, lan_dic, 512, bow=True)
    vig=TimeSeriesIG(v,lan_dic,512, bow=True)
    traindl=DataLoader(tig,batch_size=64,shuffle=True,num_workers=8)
    validdl=DataLoader(vig,batch_size=64,shuffle=True,num_workers=8)
    train(traindl, validdl, computer, optimizer, highestepoch, num_epochs=50)



def vocab_lstm_train(load=False, resplit=False):
    vocab_size=5000

    computer= LSTM_vocab(vocab_size=vocab_size)
    computer.cuda()
    computer.reset_parameters()

    lan_dic={"C":0,
             "C#":1,
             "C++":2,
             "Go":3,
             "Java":4,
             "Javascript":5,
             "Lua":6,
             "Objective-C":7,
             "Python":8,
             "Ruby":9,
             "Rust":10,
             "Shell":11}
    t, v= get_data_set(lan_dic, load=not resplit, save=resplit)
    optimizer=Adam(computer.parameters(),lr=0.001)

    id_str="vocablstm"

    if load:
        computer, optimizer, highestepoch, highestiter = load_model(computer, optimizer, 0, 0, id_str)
    else:
        highestepoch=0
        highestiter=0

    bs=64
    tig=VocabIG(t, lan_dic, vocab_size)
    vig=VocabIG(v, lan_dic, vocab_size)
    traindl=DataLoader(tig,collate_fn=pad_collate,batch_size=bs,shuffle=True,num_workers=8)
    validdl=DataLoader(vig,collate_fn=pad_collate,batch_size=bs,shuffle=True,num_workers=8)
    train(traindl, validdl, computer, optimizer, highestepoch, n_batches=len(tig)//bs, id_str=id_str, num_epochs=50,input_long=True)


def vocab_bow_train(load=False, resplit=False):
    vocab_size=5000
    max_len=100

    computer= BOW_vocab(vocab_size=vocab_size)
    computer.cuda()
    computer.reset_parameters()

    lan_dic={"C":0,
             "C#":1,
             "C++":2,
             "Go":3,
             "Java":4,
             "Javascript":5,
             "Lua":6,
             "Objective-C":7,
             "Python":8,
             "Ruby":9,
             "Rust":10,
             "Shell":11}
    t, v= get_data_set(lan_dic, load=not resplit, save=resplit)
    optimizer=Adam(computer.parameters(),lr=0.001)

    id_str="vocabbow"

    if load:
        computer, optimizer, highestepoch, highestiter = load_model(computer, optimizer, 0, 0, id_str)
    else:
        highestepoch=0

    bs=256
    tig=VocabIGpkl(t, lan_dic, vocab_size, max_len, bow=True)
    vig=VocabIGpkl(v, lan_dic, vocab_size, max_len, bow=True)
    traindl=DataLoader(tig,batch_size=bs,shuffle=False,num_workers=8)
    validdl=DataLoader(vig,batch_size=bs,shuffle=False,num_workers=8)
    train(traindl, validdl, computer, optimizer, highestepoch, n_batches=len(tig)//bs, id_str=id_str, num_epochs=50)


if __name__=="__main__":
    vocab_bow_train(load=False, resplit=False)