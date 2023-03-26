import torch
import torchvision
import os

normalize=torchvision.transforms.Normalize(
    mean=[0.5071,0.4867,0.4408],
    std=[0.2675,0.2565,0.2761]
)

train_transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32,padding=4),
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])


test_transform=torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        normalize
    ]
)

train_data=torchvision.datasets.CIFAR100(
    root=os.path.expanduser("~/.cache"),
    train=True,
    transform=train_transform,
    download=True
)

test_data=torchvision.datasets.CIFAR100(
    root=os.path.expanduser("~/.cache"),
    train=False,
    transform=test_transform,
    download=True
)


batch_size=32
epochs=20
lr=0.01
device="cuda:0"
train_loader=torch.utils.data.DataLoader(
    train_data,batch_size=batch_size,shuffle=True
)
test_loader=torch.utils.data.DataLoader(
    test_data,batch_size=batch_size,shuffle=False
)
loss=torch.nn.CrossEntropyLoss()

model=torchvision.models.resnet50('IMAGENET1K_V2').to(device)
model.fc=torch.nn.Linear(2048,100)
model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(epochs):
    model.train()
    losses=0

    for i, (x,y) in enumerate(train_loader):
        x,y=x.to(device),y.to(device)
        y_=model(x)
        l=loss(y_,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if(i+1)%5 ==0:
            print(f'loss:{(losses/5).item()}')
            losses=0
        losses+=l


    model.eval()
    with torch.no_grad():
        correct=0
        num=0

        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            y_=model(x)
            pred=torch.argmax(torch.softmax(y_,dim=1),dim=1)
            correct+=torch.sum(pred==y)
            num+=len(y)

        print('acc=',(correct/num*100).item())



