from fileinput import *
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from model import build_model
from torch.optim import Adam
from torch.autograd.variable import Variable
from os.path import abspath
from pathlib import Path
import torch

def train(data_loader, model, optimizer):
    num_epochs=10

    for epoch in range(num_epochs):
        for i,data_point in enumerate(data_loader):
            model.train()
            optimizer.zero_grad()
            input, target= data_point
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

            if i % 3==2:
                print(float(loss.item()))


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
    lan_dic={"bash":0,
             "c":1,
             "java":2,
             "python":3}
    t, v= get_data_set(lan_dic)
    optimizer=Adam(computer.parameters(),lr=0.001)
    ig=InputGen(t,lan_dic,512)
    dl=DataLoader(ig,batch_size=16,shuffle=True,num_workers=0)
    train(dl, computer, optimizer)

    save_model(computer,optimizer,0,1000,"simple")


if __name__=="__main__":
    main_train()