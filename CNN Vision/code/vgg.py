import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict


class VGG(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            
            # Stage 3
            # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
            # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),  
            nn.BatchNorm2d(256),          
            
            # Stage 4
            # TODO: convolutional layer, input channels 32, output channels 64, filter size 3
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),

            # Stage 5
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            # TODO: fully-connected layer (64->10)
            nn.Linear(512, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device, scheduler):
    for epoch in range(100):  # loop over the dataset multiple times
        scheduler.step()
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            out = net(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)
    net = VGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.08)
    train(trainloader, net, criterion, optimizer, device, scheduler)
    test(testloader, net, device)
    

if __name__== "__main__":
    main()
   
