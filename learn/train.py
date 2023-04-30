import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import time


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#prepare the data
train_data=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

#load the data
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64)

#neural network
class Cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,input):
        output=self.model(input)
        return output


model=Cifar10()

model.to(device)




#var
lr=0.01
epoches=20


#loss_function
loss_fun=nn.CrossEntropyLoss()
loss_fun.to(device)
#optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=lr)


#train
for epoch in range(epoches):
    loss_sum=0
    step=0
    start_time=time.time()
    for input,label in train_loader:

        input=input.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        output=model(input)
        loss=loss_fun(output,label)
        loss_sum+=loss
        loss.backward()
        optimizer.step()
        step+=1
    end_time=time.time()
    print(f'time is {end_time-start_time}')
    print(f'no.{epoch} train loss:{loss_sum/step}\n')

    model.eval()
    with torch.no_grad():
        loss_sum=0
        acc=0
        step=0
        for input,label in test_loader:

            input=input.to(device)
            label=label.to(device)
            output=model(input)
            loss=loss_fun(output,label)
            loss_sum+=loss
            step+=1
            output=torch.argmax(output,dim=1)
            equal=(output==label)
            acc+=equal.sum()
        print(f'no.{epoch} test loss:{loss_sum/step}')
        print(f'no.{epoch} test acc:{acc/step/64}\n')


torch.save(model,"cifar10.pth")




