import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

class Conv2d(nn.Module):
    def __init__(self):
        super(Conv2d,self).__init__()
        self.conv1=nn.Conv2d(3,6,3,1,0)

    def forward(self,x):
        x=self.conv1(x)
        return x


writer=SummaryWriter("conv2d")
test_data=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
data_loader=data.DataLoader(test_data,batch_size=64)

net=Conv2d()

step=0
for data in data_loader:
    imgs,target=data
    output=net(imgs)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images(f"input",imgs,step)
    writer.add_images(f"output",output,step)
    step+=1

writer.close()





# img_path="Pictures/erciyuan/1.jpg"
# img=Image.open(img_path)
# tensor=transforms.ToTensor()
# pil=transforms.ToPILImage()
# img_tensor=tensor(img)
#
#
# kernel=torch.ones((3,3,3,3))
# img_tensor=torch.reshape(img_tensor,(1,3,646,647))
#
#
# output=F.conv2d(img_tensor,kernel,stride=1,padding=3)
# print(output.shape)
# output=torch.reshape(output,(3,650,651))
# img=pil(output)
# img.show()





