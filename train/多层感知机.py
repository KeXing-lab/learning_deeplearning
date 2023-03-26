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
    batch_size=256,
    shuffle=True,

)

test_iter=data.DataLoader(
    data_test,
    batch_size=256,
    shuffle=False,

)

model=torch.nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
epochs=10
lr=0.1
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr)
a={}
for epoch in range(epochs):

    model.train()
    loss1=0
    num_=0
    for x,y in train_iter:
        output=model(x)
        y=y.type(torch.long)
        l=loss(output,y)
        loss1=loss1+l
        num_=num_+1
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    b=loss1.detach().numpy()
    a[epoch]=b/num_

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

plt.plot(a.keys(),a.values())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()








