from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

train_data=datasets.CIFAR10(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)

train_batch=DataLoader(train_data,batch_size=64,shuffle=True,drop_last=False)

print(train_data[0][0].shape)
writer=SummaryWriter("dataloader")

for epoch in range(2):
    step=0
    for data in train_batch:
        imgs,trget=data
        writer.add_images(f"dataloader{epoch}",imgs,step)
        step+=1

writer.close()
