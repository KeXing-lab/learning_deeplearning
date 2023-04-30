import torch
import torchvision
from PIL import Image
from torch import nn



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

test_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

img_path="C:\\Users\\Mapko\\Desktop\\OIP.jpg"
transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])
img=Image.open(img_path)
img.show()
img.convert("RGB")
img=transform(img)


img=torch.reshape(img,(1,3,32,32))

model=torch.load("cifar10.pth",map_location='cpu')
model.eval
with torch.no_grad():
    output=model(img)
    print(test_data.classes[output.argmax(1)])