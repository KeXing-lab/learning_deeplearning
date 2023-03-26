import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
import os

normalize = transforms.Normalize(
    mean=[0.5071, 0.4867, 0.4408],
    std=[0.2675, 0.2565, 0.2761]
)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
train_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"), train=True,
                                              transform=train_transform, download=True)

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])
test_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"), train=False, transform=test_transform,
                                             download=True)

batch_size = 32
epochs = 100
lr = 0.01
device = "cuda:0"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

model = torchvision.models.resnet50('IMAGENET1K_V2').to(device)
model.fc = torch.nn.Linear(2048, 100)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    losses = 0
    for i, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print("loss =", (losses / 5).item())
            losses = 0
        losses += loss

    model.eval()
    with torch.no_grad():
        correct = 0
        num = 0
        for images, target in test_loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            correct += torch.sum(pred == target)
            num += len(target)
        print("acc =", (correct / num * 100).item())