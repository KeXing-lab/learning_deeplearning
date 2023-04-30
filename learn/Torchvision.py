import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

writer=SummaryWriter("log")
transform=transforms.Compose([
    transforms.ToTensor()
])
train_data=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=transform,download=True)
test_data=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=transform,download=True)


train_batch=DataLoader(train_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
for i in range(10):
    img,target=train_data[i]
    writer.add_image("train",img,i)

writer.close()
