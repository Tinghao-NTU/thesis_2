#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch.nn.functional as F
from sklearn import metrics
import copy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
        
def feature_two(wlist, wglobal):
    state_list = []

    para_list = []
    for (_, parameters1),(_, parameters2) in zip(wlist.items(),wglobal.items()):
        para_list.append((parameters1.view(-1)- parameters2.view(-1)).cpu())

    para_list = torch.cat(para_list)
    return para_list.numpy()

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
    def get_grad(self, net):
        temp = []
        temp.append(net.conv1.weight.grad.view(-1))
        temp.append(net.conv1.bias.grad.view(-1))
        temp.append(net.conv2.weight.grad.view(-1))
        temp.append(net.conv2.bias.grad.view(-1))
        temp.append(net.fc1.weight.grad.view(-1))
        temp.append(net.fc1.bias.grad.view(-1))
        temp.append(net.fc2.weight.grad.view(-1))
        temp.append(net.fc2.bias.grad.view(-1))
        temp = torch.cat(temp)
        return temp
    def get_ParaSize(self, net):
        temp = []
        temp.append(net.conv1.weight.view(-1))
        temp.append(net.conv1.bias.view(-1))
        temp.append(net.conv2.weight.view(-1))
        temp.append(net.conv2.bias.view(-1))
        temp.append(net.fc1.weight.view(-1))
        temp.append(net.fc1.bias.view(-1))
        temp.append(net.fc2.weight.view(-1))
        temp.append(net.fc2.bias.view(-1))
        temp = torch.cat(temp)
        return len(temp)      

    def train(self, net, prob = 0):
        global_model = copy.deepcopy(net)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        delta_F_local = torch.zeros(self.get_ParaSize(net)).to(self.args.device)
        count = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                #print(np.shape(images))
                log_probs = net(images)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if iter == 0:
                F_global = sum(epoch_loss) / len(epoch_loss)
        
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            #print(np.shape(images))
            log_probs = net(images)
            loss = F.nll_loss(log_probs, labels)
            batch_loss.append(loss.item())
            delta_F_local += self.get_grad(net)
            count += 1  
        temp_loss = sum(batch_loss)/len(batch_loss)
        F_local= temp_loss
        delta_F_local = delta_F_local.detach().cpu()/count
        delta_F_global = feature_two(net.state_dict(),global_model.state_dict())/(5 * self.args.lr)
        
        return net.state_dict(),F_local,delta_F_local, F_global, delta_F_global

    
    def test(self, net):
        net_g.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = self.ldr_train
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        if self.args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        return accuracy
