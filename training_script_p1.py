import torch
import logging
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import pandas as pd
from datetime import datetime


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 128
path = "/scratch/tor213/DLS/"
data_path = "/scratch/tor213/"

trainset = torchvision.datasets.CIFAR10(root=f'{data_path}data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=f'{data_path}data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img.to("cpu") / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainig_batch_size,
                                            shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=testing_batch_size,
                                            shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader

class Net(nn.Module):
    def __init__(self, use_dropout=False, num_classes=10):
        super().__init__()
        self.use_dropout = use_dropout
        self.flatten = nn.Flatten()
        self.input_dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(32 * 32 * 3, 1_000)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1_000, 1_000)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1_000, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        if self.use_dropout:
            x = self.input_dropout(x)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return x

# function to create all the optimizers that we need
def get_optimizer(model, name):
    if name == 'AdaGrad':
        return optim.Adagrad(params=model.parameters(), lr=0.01, lr_decay=0, weight_decay=1e-4, initial_accumulator_value=0, eps=1e-10)
    elif name == 'RMSProp':
        return optim.RMSprop(params=model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=1e-4, momentum=0, centered=False)
    elif name == 'RMSProp+Nesterov':
        return optim.NAdam(params=model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, momentum_decay=0.004)
    elif name == 'AdaDelta':
        return optim.Adadelta(params=model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=1e-4)
    elif name == 'Adam':
        return optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    else:
        return None

epochs = 4
optimizers = ['AdaGrad', 'RMSProp', 'RMSProp+Nesterov', 'AdaDelta', 'Adam'] 
# Train the network without dropout
for opt_name in optimizers:
    print(f"\n\nTraining using optimizer: {opt_name}")
    net = Net().to(device)
    train(net, optimizer=get_optimizer(net, opt_name), trainloader=trainloader, testloader=testloader, epochs=epochs, name=f"{opt_name.replace('+', '_')}")

# add dropout to the model
for opt_name in optimizers:
    print(f"\n\nTraining using optimizer: {opt_name}")
    net = Net(use_dropout=True).to(device)
    train(net, optimizer=get_optimizer(net, opt_name), trainloader=trainloader, testloader=testloader, epochs=epochs, name=f"dropout_{opt_name.replace('+', '_')}")