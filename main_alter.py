from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from models import transfer
import torchvision.models as models

import logging
import time
import random
import numpy as np
import os

from models.resnet import resnet50, lowrank_resnet50_conv1x1
from vgg import *
from lowrank_vgg import LowRankVGG, FullRankVGG, VanillaVGG19, LowRankVGG19, LRVGG19Residule
from resnet_cifar10 import *

from ptflops import get_model_complexity_info

best_acc = 0  # best test accuracy

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

# trainset = datasets.CIFAR100(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=256, shuffle=True, num_workers=4,
#                 pin_memory=True)
# testset = datasets.CIFAR100(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=256, shuffle=False, num_workers=4,
#     pin_memory=True)



# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')


RANK_FACTOR = 8
# Model
print('==> Building model..')
net = transfer.LowRankResNet50()
# net = LowRankResNet50()
net = net.to(device)

print("HERE IS LOWRANKNET")

# net_vanilla = models.resnet50()
# net_vanilla = net_vanilla.to(device)
# net_vanilla.eval()
# print("HERE IS NET VANILLA")
# print(net_vanilla)

net_vanilla = transfer.ResNet50(pretrained=False)
# # net_vanilla.load_state_dict(torch.load('/content/gdrive/MyDrive/ResNet50.pth'))
net_vanilla = net_vanilla.to(device)
net_vanilla.eval()

# cudnn.benchmark = True

print("!!!! VANILLA Resnet50 : {}".format(net_vanilla))
# print("@@@ Vanilla VGG19 : {}".format(net_vanilla))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
#                         milestones=[150, 250], gamma=0.1)

# optimizer_vanilla = optim.Adam(net_vanilla.parameters(), lr=1e-4)
optimizer_vanilla = optim.SGD(net_vanilla.parameters(), lr=0.05,
                      momentum=0.9, weight_decay=5e-4)
# scheduler_vanilla = torch.optim.lr_scheduler.OneCycleLR(optimizer_vanilla, max_lr=0.05, steps_per_epoch=len(trainloader), epochs=30)

# scheduler_vanilla = torch.optim.lr_scheduler.MultiStepLR(optimizer_vanilla, 
#                         milestones=[20, 30], gamma=0.1)

def decompose_weights(model, low_rank_model, rank_factor):
    # SVD version
    reconstructed_aggregator = []
    
    for item_index, (param_name, param) in enumerate(model.state_dict().items()):
        if len(param.size()) == 4 and item_index not in range(0, 2) and 'shortcut' not in param_name and 'downsample' not in param_name:
            # resize --> svd --> two layer
            # print(param.size())
            param_reshaped = param.view(param.size()[0], -1)
            rank = min(param_reshaped.size()[0], param_reshaped.size()[1])
            # print(rank)
            u, s, v = torch.svd(param_reshaped)

            sliced_rank = int(rank/rank_factor)
            u_weight = u * torch.sqrt(s) # alternative implementation: u_weight_alt = torch.mm(u, torch.diag(torch.sqrt(s)))
            v_weight = torch.sqrt(s) * v # alternative implementation: v_weight_alt = torch.mm(torch.diag(torch.sqrt(s)), v.t())
            # sanity check: print("dist u u_alt:{}, dist v v_alt: {}".format(torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt)))
            # print("## v weight size: {}, v weight alt size: {}".format(v_weight.size(), v_weight_alt.size()))
            #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt)))
            #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.equal(u_weight, u_weight_alt), torch.equal(v_weight.t(), v_weight_alt)))
            #print("dist param impl: {}, dist param impl alt: {}".format(torch.dist(param_reshaped, torch.mm(u_weight_alt, v_weight_alt)), torch.dist(param_reshaped, torch.mm(u_weight, v_weight.t()))))
            #print("dist: {}".format(torch.dist(torch.mm(u_weight_alt, v_weight_alt), torch.mm(u_weight, v_weight.t()))))


            u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]

            approx = torch.mm(u_weight_sliced, v_weight_sliced.t())
            
            res = param_reshaped - approx

            u_weight_sliced_shape, v_weight_sliced_shape = u_weight_sliced.size(), v_weight_sliced.size()

            # print(u_weight_sliced_shape)
            model_weight_v = u_weight_sliced.view(u_weight_sliced_shape[0],
                                                  u_weight_sliced_shape[1], 1, 1)
            
            model_weight_u = v_weight_sliced.t().view(v_weight_sliced_shape[1], 
                                                      param.size()[1], 
                                                      param.size()[2], 
                                                      param.size()[3])
            model_res = res.view(param.size())

            reconstructed_aggregator.append(model_weight_u)
            reconstructed_aggregator.append(model_weight_v)
            reconstructed_aggregator.append(model_res)    
        else:
            reconstructed_aggregator.append(param)
            
            
    model_counter = 0
    reload_state_dict = {}
    for item_index, (param_name, param) in enumerate(low_rank_model.state_dict().items()):
        print("#### {}, {}, recons agg: {}， param: {}".format(item_index, param_name, 
                                                                               reconstructed_aggregator[model_counter].size(),
                                                                              param.size()))
        assert (reconstructed_aggregator[model_counter].size() == param.size())
        reload_state_dict[param_name] = reconstructed_aggregator[model_counter]
        model_counter += 1
    
    low_rank_model.load_state_dict(reload_state_dict)
    return low_rank_model


# Training
def train(epoch, model, optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(transfer.trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            logger.info("Train @ Epoch: {}, {}/{}, Loss: {:.3f}, Acc: {:.3f}".format(
                epoch, batch_idx, len(transfer.trainloader), train_loss/(batch_idx+1), 100.*correct/total))


def test(epoch, model):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(transfer.testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        logger.info("Test @ Epoch: {}, Loss: {:.4f}, Acc: {:.2f}".format(
                epoch, test_loss/total, 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

TOTAL = 200
WARM_UP = 200
scheduler = optim.ReduceLROnPlateau(optimizer_vanilla, 'min')
for epoch in range(start_epoch, start_epoch+TOTAL):
    #for param_index, (param_name, param) in enumerate(net.named_parameters()):
    #    print("!!!! Param idx: {}, param name: {}, param size: {}".format(
    #            param_index, param_name, param.size()))
    if epoch in range(WARM_UP):
        print("!!!!! Warm-up epoch: {}".format(epoch))
        transfer.fit_epoch(net_vanilla, transfer.trainloader, criterion, optimizer_vanilla)
        val_loss, _ = transfer.validate_epoch(net_vanilla, transfer.testloader, criterion)
        # train(epoch, model=net_vanilla, optimizer=optimizer_vanilla)
        # test(epoch, model=net_vanilla)
        scheduler.step(val_loss)
    elif epoch == WARM_UP:
        print("!!!!! Switching to low rank model, epoch: {}".format(epoch))
        net = decompose_weights(model=net_vanilla, 
            low_rank_model=net, rank_factor=RANK_FACTOR)
        transfer.validate_epoch(net, transfer.testloader, criterion)

        optimizer = optim.SGD(net.parameters(), lr=0.05,
                              momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
        #                         milestones=[10, 15], gamma=0.1)

        # freeze the residual layers
        for param_index, (param_name, param) in enumerate(net.named_parameters()):
            #print("!!!! Param idx: {}, param name: {}, param size: {}".format(
            #        param_index, param_name, param.size()))
            if "_res" in param_name:
                param.requires_grad = False

        transfer.fit_epoch(net, transfer.trainloader, criterion, optimizer) # why different?
        transfer.validate_epoch(net, transfer.testloader, criterion)
        # train(epoch, model=net, optimizer=optimizer)
        # test(epoch, model=net)
        for group in optimizer.param_groups:
            print("@@@@@ Epoch: {}, Lr: {}".format(epoch, group['lr']))
        # scheduler.step()
    else:
        if epoch % 5 == 0:
            # un-freeze the residual layers
            for param_index, (param_name, param) in enumerate(net.named_parameters()):
                #print("!!!! Param idx: {}, param name: {}, param size: {}".format(
                #        param_index, param_name, param.size()))
                if "_res" in param_name:
                    param.requires_grad = True
        else:
            # freeze the residual layers
            for param_index, (param_name, param) in enumerate(net.named_parameters()):
                #print("!!!! Param idx: {}, param name: {}, param size: {}".format(
                #        param_index, param_name, param.size()))
                if "_res" in param_name:
                    param.requires_grad = False

        print("!!!!! low rank training, epoch: {}".format(epoch))
        for group in optimizer.param_groups:
            print("@@@@@ Epoch: {}, Lr: {}".format(epoch, group['lr']))

        transfer.fit_epoch(net, transfer.trainloader, criterion, optimizer) # why different?
        transfer.validate_epoch(net, transfer.testloader, criterion)
        # scheduler.step()