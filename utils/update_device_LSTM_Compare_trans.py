import torch.nn as nn
import torch.nn.functional as F

import torch
from sklearn.cluster import KMeans
import numpy as np

import copy
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
from models.UpdateCompare import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
def feature_cal(wlist, wglobal):
    state_list = []
    for i in range(100):
        para_list = []
        for (_, parameters1),(_, parameters2) in zip(wlist[i].items(),wglobal.items()):
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 1,dim = 0).cpu().numpy().item())
        state_list.append(torch.tensor(para_list).view(1,-1))
#     para_list = torch.cat(para_list)
    return torch.cat(state_list,0)

def feature_two(wlist, wglobal):
    state_list = []

    para_1 = []
    para_2 = []
    for (_, parameters1),(_, parameters2) in zip(wlist.items(),wglobal.items()):
        para_1.append(parameters1.view(-1).cpu())
        para_2.append(parameters2.view(-1).cpu())
    para_1 = torch.cat(para_1)
    para_2 = torch.cat(para_2)

    para_list = torch.norm((para_1- para_2), p = 1)
    return para_list.numpy().item()

def get_state(clusteded_dic,state,i):
    i = '%d'%(i)
    device_idx = np.array(clusteded_dic[i])
    device_num = len(np.array(clusteded_dic[i]))
    return state[device_idx].numpy()#
def permute_device(action, clusteded_dic, cluster_index):
    cluster_index = '%d'%(cluster_index)
    real_index = action
    device_list = clusteded_dic[cluster_index]
    selected_device = device_list[real_index]
    index_list = np.delete(device_list,real_index)
    device_list = np.append(index_list,selected_device).tolist()
    clusteded_dic[cluster_index] = device_list
    return clusteded_dic
def cluster_devices(wlist):    
    dist_mx = []
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    
    for i in range(100):
        temp = copy.deepcopy(wlist[i])
        para_list1 = temp['fc2.weight']
        dist_mx.append(para_list1.view(1,-1))
    dist_mx = torch.cat(dist_mx,0)
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(dist_mx.cpu())
    for i in range(10):
        class_index[i] = np.where(kmeans.labels_ == i)[0].tolist()
    return class_index


def local_update(args,dataset_train,dataset_test,dict_users,idxs_users,epoch,net_glob,rou,beta,delta,acc_list = None,w_list = None, name = None,preset  = None, seed = None):
    w_locals_temp = []
    loss_locals = []
    reward_n = []
    if preset is None:
        preset  = 99


        
    w_global_previous = copy.deepcopy(net_glob.state_dict())
    global_grad_loss = [0] * 100
    tmp = 0
    for idx in idxs_users:
        net_temp = copy.deepcopy(net_glob)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[0][idx])
        w,F_local_i,delta_F_local_i, F_global_i, delta_F_global_i = local.train(net=copy.deepcopy(net_glob).to(args.device))
        result = w['fc2.weight'].view(1,-1)
        w_locals_temp.append(copy.deepcopy(w))
        rou[idx]  = abs(F_local_i-F_global_i)/(feature_two(w, w_global_previous))
        beta[idx] = torch.norm((delta_F_local_i-delta_F_global_i), p = 1)/(feature_two(w, w_global_previous))
        global_grad_loss[idx] = delta_F_global_i
        tmp += delta_F_global_i
        #net_temp.load_state_dict(w)
        #acc_train, loss_train = test_img(net_temp, dataset_test, args)
        #reward_n.append(pow(coeff , acc_train.numpy().item()-preset)-1) 

        if epoch == 0:
            w_list.append(w)

        else:
            w_list[idx] = copy.deepcopy(w)
    sum_global_loss = tmp/len(idxs_users)
#     print(global_grad_loss[idx])
#     print(sum_global_loss)
    for idx in idxs_users:
        delta[idx] = np.linalg.norm((global_grad_loss[idx]-sum_global_loss), ord=1) 
    
    
    w_glob = FedAvg(w_locals_temp)
    state_matrix = []

    net_glob.load_state_dict(w_glob)

    # print loss
                
#         acc_test, loss_test = test_img(net_glob, dataset_test, args)     
    acc_train, loss_train = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_train)
    
    print('Round {:3d}, Training accuracy {:.3f}'.format(epoch, acc_train))

    
    if name is not None:
        file_handle=open(name,mode='a')
        file_handle.write(str(acc_train.numpy().item()))
        file_handle.write('\n')
        file_handle.close()


    if acc_train < preset :
        done_n = [0] * 10
    else:
        done_n = [1] * 10

    return state_matrix,net_glob, acc_list,w_list,done_n,rou,beta,delta

# def get_state(clusteded_dic,state,i):
#     i = '%d'%(i)
#     device_idx = np.array(clusteded_dic[i])
#     return torch.cat(((state[device_idx],state[-1].view(1,-1))),0).view(-1).cpu().detach().numpy()