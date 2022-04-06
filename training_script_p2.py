import torch
import logging
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


data_path = "/scratch/tor213/data"
path = "/scratch/tor213/DLS/"

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                    download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                    download=True, transform=transform)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def test_network(net, criterion, testloader, epoch=None, showimages=False, print_freq=100):
    correct = 0
    total = 0
    meters = {name: AverageMeter() for name in ['step', 'data', 'loss', 'acc']}
    end = time.time()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            meters['data'].update(time.time() - end)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Calculate running average of accuracy
            pred = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (pred.to(device) == labels.to(device).data).sum().item()
            acc = correct / total
            meters['acc'].update(float(acc), images.size(0))
            # measure accuracy and record loss
            loss = criterion(outputs, labels)
            meters['loss'].update(float(loss), images.size(0))
            # measure elapsed time
            meters['step'].update(time.time() - end)
            if i % print_freq == 0 or i == len(testloader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Acc {meters[acc].val:.3f} ({meters[acc].avg:.3f})\t'
                             .format(
                                 epoch + 1, i, len(testloader),
                                 phase='EVALUATING',
                                 meters=meters)) 
                print(report)
            end = time.time()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')
    if showimages: 
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
    return meters

def train(net, optimizer, trainloader, testloader, epochs=1, criterion=nn.CrossEntropyLoss(), lr=0.001, momentum=0.9, print_freq=100, name=''):
    df_columns = ['train_step','train_data','train_loss','train_acc','test_step','test_data','test_loss','test_acc','total_train_step_time','total_train_data_time','total_test_step_time','total_test_data_time']
    results_df = pd.DataFrame(columns=df_columns)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Start Training {name}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch #{epoch + 1}")
        meters = {name: AverageMeter() for name in ['step', 'data', 'loss', 'acc']}
        correct = 0.
        total = 0.
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # measure data loading time
            meters['data'].update(time.time() - end)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # measure elapsed time
            meters['step'].update(time.time() - end)

            # Calculate running average of accuracy
            pred = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (pred.to(device) == labels.to(device).data).sum().item()
            acc = correct / total
            meters['acc'].update(float(acc), inputs.size(0))
            # measure accuracy and record loss
            meters['loss'].update(float(loss), inputs.size(0))
            if i % print_freq == 0 or i == len(trainloader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.4f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.4f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Acc {meters[acc].val:.3f} ({meters[acc].avg:.3f})\t'
                             .format(
                                 epoch + 1, i, len(trainloader),
                                 phase='TRAINING',
                                 meters=meters)) 
                print(report)
            end = time.time()
        test_meters = test_network(net, criterion, testloader, epoch)

        print(str('\nResults - Epoch: {0}\n'
                'Training Loss {train[loss].avg:.4f} \t'
                'Training Accuracy {train[acc].avg:.4f} \t'
                'Training Step Time  {train[step].sum:.4f} \t'
                'Training Data Time  {train[data].sum:.4f} \t'
                'Validation Loss {val[loss].avg:.4f} \t'
                'Validation Accuracy {val[acc].avg:.4f} \n'
                .format(epoch + 1, train=meters, val=test_meters)))
        train_df = {"train_"+k: v.avg for k, v in meters.items()}
        test_df = {"test_"+k: v.avg for k, v in test_meters.items()}
        epoch = {**train_df, **test_df}
        epoch["total_train_step_time"] = meters["step"].sum
        epoch["total_train_data_time"] = meters["data"].sum
        epoch["total_test_step_time"] = test_meters["step"].sum
        epoch["total_test_data_time"] = test_meters["data"].sum
        results_df.loc[len(results_df)] = [v for _, v in epoch.items()]
        
    print(f'Finished Training {name}\n')
    results_df.to_csv(f"{path}{name}.csv", index=False)
    return results_df


def get_dataloaders(trainig_batch_size=128, testing_batch_size=100):
    transform = transforms.Compose(
        [transforms.RandomCrop(size=32, padding=4),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.44655), (0.2023, 0.1994, 0.20105)) ])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainig_batch_size,
                                            shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=testing_batch_size,
                                            shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

max_batch_size = 2**22
print(f"Starting training with {torch.cuda.device_count()} GPUs")
b = 32  # Initial batch size
num_epochs = 2  # The first epoch is to bring things to the cache and stuff
df_columns = ['gpu_num', 'batch_size', 'train_step','train_data','train_loss','train_acc','test_step','test_data','test_loss','test_acc','total_train_step_time','total_train_data_time','total_test_step_time','total_test_data_time']
results = pd.DataFrame(columns=df_columns)
while b < max_batch_size:
    try:
        print(f"Training with batch size: {b} and {torch.cuda.device_count()} GPUs")
        net = ResNet18().to(device)
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            # For example, for N = 3, the expected dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            net = nn.DataParallel(net)
        trainloader, testloader = get_dataloaders(trainig_batch_size=b*torch.cuda.device_count())  # multiply the batch size times the number of GPUs 
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train_df = train(net, optimizer, epochs=num_epochs, trainloader=trainloader, testloader=testloader, name=f'gpu_{torch.cuda.device_count()}_batch_size_{b}')
        results_from_second_epoch = train_df.iloc[-1].to_dict()
        results_from_second_epoch['batch_size'] = b
        results_from_second_epoch['gpu_num'] = torch.cuda.device_count()
        results = results.append(results_from_second_epoch, ignore_index=True)
    except RuntimeError as e:
        b = b // 4
        print(f"LIMIT REACHED: The limit for {torch.cuda.device_count()} GPU(s) is: batch_size={b:,} with an effective batch of {b*torch.cuda.device_count():,}")
        break
    if b >= max_batch_size:
        break
    b *= 4
print(f"Now we know that the limit for {torch.cuda.device_count()} GPU(s) is: batch_size={b:,} with an effective batch of {b*torch.cuda.device_count():,}")
results.to_csv(f"{path}gpu_{torch.cuda.device_count()}_results.csv", index=False)
