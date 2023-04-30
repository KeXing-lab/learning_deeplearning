import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image



class vgg16(nn.Module):
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


class Max_pool(nn.Module):
    def __init__(self):
        super(Max_pool,self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        x=self.maxpool(x)
        return x


# writer=SummaryWriter("maxpool")
# test_data=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
# data_loader=data.DataLoader(test_data,batch_size=64)

# net=vgg16()
#
#
#
# torch.save(net,"vgg16.pth")
# torch.save(net.state_dict(),"vgg16para.pth")
#
# model=vgg16()
# model.load_state_dict(torch.load("vgg16para.pth"))
# print(model)




# optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
# loss=nn.CrossEntropyLoss()

# for epoch in range(20):
#     sum_loss=0
#     for data in data_loader:
#         imgs,target=data
#         output=net(imgs)
#         optimizer.zero_grad()
#         l=loss(output,target)
#         l.backward()
#         optimizer.step()
#         sum_loss+=l
#     print(f'loss is {sum_loss}')



model=torchvision.models.vgg16(pretrained=True)
model.classifier.add_module("tt",nn.Linear())
model.classifier[6]=nn.Linear()
