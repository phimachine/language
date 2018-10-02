from fileinput import *
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from model import computer
from torch.optim import Adam
from torch.autograd.variable import Variable

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


def main():
    computer.cuda()
    computer.reset_parameters()
    t, v, lan_dic= get_data_set()
    optimizer=Adam(computer.parameters(),lr=0.001)
    ig=InputGen(t,lan_dic,512)
    dl=DataLoader(ig,batch_size=16,shuffle=True,num_workers=0)
    train(dl, computer, optimizer)


if __name__=="__main__":
    main()