import torch.nn as nn
import torch.nn.functional as F

import torch
from sklearn.cluster import KMeans
import numpy as np
from opti_FL_test import optimization_bf
import copy
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
from models.UpdateProb import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
def generate_shadow_fading(mean,sigma,size):
    sigma=pow(10,sigma/10)
    mean=pow(10,mean/10)
    m = np.log(pow(mean,2)/np.sqrt(pow(sigma,2)+pow(mean,2)))
    sigma= np.sqrt(np.log(pow(sigma,2)/pow(mean,2)+1))
    np.random.seed(0)
    Lognormal_fade=np.random.lognormal(m,sigma,size)
    return Lognormal_fade


def feature_cal(wlist, wglobal):
    state_list = []
    for i in range(100):
        para_list = []
        for (_, parameters1),(_, parameters2) in zip(wlist[i].items(),wglobal.items()):
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 1,dim = 0).cpu().numpy().item())
        state_list.append(torch.tensor(para_list).view(1,-1))
#     para_list = torch.cat(para_list)
    return torch.cat(state_list,0)
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
    



def local_update(args,dataset_train,dataset_test,dict_users,idxs_users,epoch,net_glob,acc_list = None,w_list = None, name = None,preset  = None, coeff = None,gtk = None, prob = 0):
    w_locals_temp = []
    loss_locals = []
    all_clients = 100
    n = all_clients
    np.random.seed(1)
    random_num = np.random.random(all_clients)
    dist = (10+1990*random_num)*0.001
    dist1 = 50*0.001
    dist0 = 10*0.001
    plm = 128.1 + 37.6*np.log10(dist)
    rd_number = generate_shadow_fading(0,8,n)
    PathLoss_User_BS=plm+rd_number
    g = pow(10,(-PathLoss_User_BS/10))
    N0 =-174
    N0 =pow(10,(N0/10))/1e3
    i = np.array([i for i in range(0,all_clients)])
    C = np.random.random(n) * 9e4 + 1e4
    B = 1*1e6
    f = np.random.uniform(0.1e9,2e9,100)
    p = np.random.uniform(0.001,0.19952,100)
    k = 10000
    if args.dataset == 'mnist':
        s = 2488888
        D = np.array([60000/all_clients] * all_clients)
    if args.dataset == 'fashion':
        s = 438888
    else:
        s = 4894444
        D = np.array([50000/all_clients] * all_clients)
    
    E,T,f_list, p_list, b_list, g_list = optimization_bf(f, p, g, C, D, k, s, B,idxs_users)
    reward_n = -(E + 1 * T)
    if preset is None:
        preset  = 99
    if coeff is None:
        coeff = 1.2

    if epoch ==0:
        acc_list = [-100000]
        w_list = [] * 100
        gtk = [1] * 100
    for idx in idxs_users:
        net_temp = copy.deepcopy(net_glob)
        idx_str = '%d'%(idx)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx_str])
        if prob != 0:
            w, loss, gtk_temp = local.train(net=copy.deepcopy(net_glob).to(args.device), prob = prob)
            gtk[idx] = gtk_temp
        else:
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        result = w['fc2.weight'].view(1,-1)
        w_locals_temp.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))

        #net_temp.load_state_dict(w)
        #acc_train, loss_train = test_img(net_temp, dataset_test, args)
        #reward_n.append(pow(coeff , acc_train.numpy().item()-preset)-1) 

        #w_list[idx] = copy.deepcopy(w)

    w_glob = FedAvg(w_locals_temp)
    state_matrix = []

    net_glob.load_state_dict(w_glob)

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
#         acc_test, loss_test = test_img(net_glob, dataset_test, args)     
    acc_train, loss_train = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_train)
    if name is not None:
        file_handle=open(name,mode='a')
        file_handle.write(str(acc_train.numpy().item()))
        file_handle.write('\n')
        file_handle.close()

    if acc_train < preset :
        done_n = [0] * 10
    else:
        done_n = [1] * 10
    return state_matrix,net_glob, acc_list,w_list,reward_n,done_n,gtk

# def get_state(clusteded_dic,state,i):
#     i = '%d'%(i)
#     device_idx = np.array(clusteded_dic[i])
#     return torch.cat(((state[device_idx],state[-1].view(1,-1))),0).view(-1).cpu().detach().numpy()