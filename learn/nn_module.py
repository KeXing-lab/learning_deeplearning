import torch
import torch.nn as nn

class Mymodule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        output=x+1
        return output

net=Mymodule()
input=torch.tensor()
output=net(input)
print(output)