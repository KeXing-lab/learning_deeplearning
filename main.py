import torch
import torchvision
import d2l
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
import os
import numpy






def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device=next(iter(net.parameters())).device
        metric=d2l.Animator(2)
        with torch.no_grad():
            for x,y in data_iter:
                if isinstance(x,list):
                    x=[x_.to(device) for x_ in x]
                else:
                    x=x.to(device)
                y=y.to(device)
                metric.add(d2l.accuracy(net(x),y),y.numel())
            return metric[0]/metric[1]

def train(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
        print("training on ",device)
        net.to(device)
        optimizer =torch.optim.SGD(net.parameters(),lr=lr)
        loss=nn.CrossEntropyLoss()
        animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                              legend=['train loss','train acc','test acc'])
        timer,num_batches = d2l.Timer(),len(train_iter)
        for epoch in range(num_epochs):
            metric=d2l.Accumulator(3)
            net.train()
            for i ,(x,y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                x,y = x.to(device),y.to(device)
                y_hat=net(x)
                l=loss(y_hat,y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l*x.shape[0], d2l.accuracy(y_hat,y),x.shape[0])
                timer.stop()
                train_l=metric[0]/metric[2]
                train_acc=metric[1]/metric[2]
                if (i+1)%(num_batches // 5) ==0 or i==num_batches-1:
                    animator.add(epoch+(i+1)/num_batches,
                                 (train_l,train_acc,None))
            test_acc=evaluate_accuracy_gpu(net,test_iter)
            animator.add(epoch+1,(None,None,test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f},test acc {test_acc:.3f}')
        print(f'{metric[2]*num_epochs/timer.sum():.1f} examples/sec on {device}')

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.con1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.con2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1x1conv:
            self.con3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.con3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

    def forward(self,x):
        y=F.relu(self.bn1(self.con1(x)))
        y=self.bn2(self.con2(y))
        if self.con3:
            x=self.con3(x)
        y+=x
        return F.relu(y)

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cifar100_train=torchvision.datasets.CIFAR100(
    root="../data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

cifar100_test=torchvision.datasets.CIFAR100(
    root="../data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

trainloader=torch.utils.data.DataLoader(
    cifar100_train,
    batch_size=100,
    shuffle=True,
    num_workers=2

)

testloader=torch.utils.data.DataLoader(
    cifar100_test,
    batch_size=100,
    shuffle=False,
    num_workers=2
)

x,y = next(iter(trainloader))
print(x.size())
print(y.size())

b1=nn.Sequential(
    nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))

net=nn.Sequential(
    b1,b2,b3,b4,b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),nn.Linear(512,100)
)

# train_iter,test_iter=d2l.load_data_fashion_mnist(256,resize=96)
# d2l.train_ch6(net,trainloader,testloader,100,0.05,d2l.try_gpu())
# d2l.plt.show()




