import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
import os
from matplotlib import pyplot as plt

data_train=torchvision.datasets.FashionMNIST(
    root='../data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

data_test=torchvision.datasets.FashionMNIST(
    root='../data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_iter=data.DataLoader(
    data_train,
    batch_size=32,
    shuffle=True,

)

test_iter=data.DataLoader(
    data_test,
    batch_size=32,
    shuffle=False,

)

model=torch.nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,10),

)
epochs=50
lr=0.03
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(epochs):

    model.train()
    for x,y in train_iter:
        output=model(x)
        y=y.type(torch.long)

        l=loss(output,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


    model.eval()
    num=0
    right=0
    with torch.no_grad():
        for x,y in test_iter:
            out=model(x)
            pred=torch.argmax(torch.softmax(out,dim=1),dim=1)
            num+=len(pred)
            right+=torch.sum(pred==y)

    print(f'epoch:{epoch} acc:{(right/num*100).item()}')








