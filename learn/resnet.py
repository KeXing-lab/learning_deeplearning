import torch
import torchvision
import torchvision.transforms as T
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

transform=T.Compose([
    T.ToTensor(),
    #T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
])
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=transform,download=True)
test_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=transform,download=True)

batch_size=64
lr=0.01
epoches=2
train_loader=data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader=data.DataLoader(test_data,batch_size=batch_size,shuffle=False)
writer=SummaryWriter("resnet")
loss_fun=torch.nn.CrossEntropyLoss()



model=torchvision.models.resnet50(weights='IMAGENET1K_V2')
model.add_module("tt",torch.nn.Linear(1000,10))
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

model.to(device)
loss_fun.to(device)



for epoch in range(epoches):
    model.train()
    step=0
    loss_sum=0
    for input,target in train_loader:
        input=input.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(input)
        loss=loss_fun(output,target)
        loss_sum+=loss
        loss.backward()
        optimizer.step()
        step+=1

    print(f'\nno.{epoch+1} train loss is {loss_sum/step}\n')
    writer.add_scalar('train_loss',loss_sum/step,epoch+1)



    model.eval()
    loss_sum=0
    step=0
    acc=0
    with torch.no_grad():
        for input,target in test_loader:
            input=input.to(device)
            target=target.to(device)
            output=model(input)
            loss=loss_fun(output,target)
            loss_sum+=loss
            acc+=((torch.argmax(output,dim=1)==target).sum())
            step+=1

        print(f'no.{epoch+1} test loss is {loss_sum/step}')
        print(f'no.{epoch+1} test acc is {acc/(step*batch_size)*100}%\n')
        writer.add_scalar('test_loss', loss_sum / step, epoch + 1)
        writer.add_scalar('test_acc', acc/(step*batch_size), epoch + 1)

writer.close()





