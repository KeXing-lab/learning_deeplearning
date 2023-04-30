from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
import cv2


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.image_path=os.listdir(self.path)

    def __getitem__(self, item):
        img_name=self.image_path[item]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.image_path)








img_path="Pictures/erciyuan/a.jpg"
img=Image.open(img_path)

compose=transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

img_=compose(img)
print(img)

writer=SummaryWriter("logs")
writer.add_image("i",img_)

writer.close()
