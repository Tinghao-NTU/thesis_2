import torch.nn as nn
import torch.nn.functional as F
# import gym
import torch
from sklearn.cluster import KMeans
import numpy as np
from opti_FL import optimization_bf
import copy
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
from models.Update import LocalUpdate
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
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 2,dim = 0).cpu().numpy().item())
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
        para_list1 = temp['fc2.bias']
        dist_mx.append(para_list1.view(1,-1))
    dist_mx = torch.cat(dist_mx,0)
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(dist_mx.cpu())
    for i in range(10):
        class_index[i] = np.where(kmeans.labels_ == i)[0].tolist()
    return class_index

all_clients = 100
n = all_clients
np.random.seed(0)
random_num = np.random.random(all_clients)
dist = (50+950*random_num)*0.001 #250*1.414*np.random.random(50);# 在500*500正方形区域内
dist1 = 50*0.001
dist0 = 10*0.001
plm = 128.1 + 37.6*np.log10(dist)
rd_number = generate_shadow_fading(0,8,n)
PathLoss_User_BS=plm+rd_number
g_list = pow(10,(-PathLoss_User_BS/10))
N0 =-174
N0 =pow(10,(N0/10))/1e3
i = np.array([i for i in range(0,all_clients)])
C_list = (9e3/100)*i + 1e3
np.random.seed(0)
np.random.shuffle(C_list)
B = 20*1e6
num_clients = 50
n = num_clients
f_max = 2*1e9
f = np.ones(num_clients) * f_max
f_min = 0.2*1e9
p_max = 0.19952
p_min = 0.01
p = np.ones(num_clients) *p_max
delta = 0.1
xi = 0.1 
epsilon = 0.001 
s = 4894444
alpha = 2e-28 
D = 500    
k = 1e-28
E =  0.3*np.ones(num_clients)
T0 = 100
b_init= np.ones(num_clients) * (B/num_clients)
b = b_init
eta = 0.5
gamma = 1
l = 1


count = 0 
ratio_b  = 0
tola = 0.1



def local_update(args,dataset_train,dataset_test,dict_users,idxs_users,epoch,net_glob,acc_list = None,w_list = None, name = None,preset  = None, coeff = None):
    w_locals_temp = []
    loss_locals = []
    g = g_list[idxs_users]
    C = C_list[idxs_users]
    if args.dataset == 'mnist':
        s = 2488888
    if args.dataset == 'fashion':
        s = 438888
    else:
        s = 4894444

    all_clients = 100
    n = all_clients
    np.random.seed(0)
    random_num = np.random.random(all_clients)
    dist = (50+950*random_num)*0.001  
    
    T_result,E_result = optimization_bf(len(idxs_users), g, N0, C, D, k, args.s, B)
    reward_n = -(T_result + E_result)
    
    if preset is None:
        preset  = 99
    if coeff is None:
        coeff = 1.2

    if epoch ==0:
        acc_list = [-100000]
        w_list = []
    for idx in idxs_users:
        idx = int(idx)
        net_temp = copy.deepcopy(net_glob)
        idx_str = '%d'%(idx)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx_str])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        w_locals_temp.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))


        if epoch == 0:
            w_list.append(w)

        else:
            w_list[idx] = copy.deepcopy(w)

    w_glob = FedAvg(w_locals_temp)
    state_matrix = feature_cal(w_list, w_glob)

    net_glob.load_state_dict(w_glob)

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
#         acc_test, loss_test = test_img(net_glob, dataset_test, args)     
    acc_train, loss_train = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_train)
    print('Round {:3d}, Average loss {:.3f}, Training accuracy {:.3f}'.format(epoch, loss_avg,acc_train))
    if name is not None:
        file_handle=open(name,mode='a')
        file_handle.write(str(acc_train.numpy().item()))
        file_handle.write('\n')
        file_handle.close()




    if acc_train < preset :
        done_n = [0] * 10
    else:
        done_n = [1] * 10
    if epoch == 0 :
        #clusteded_dic = cluster_devices(w_list)
        return state_matrix,net_glob, acc_list,w_list,w_list
    else:
        return state_matrix,net_glob, acc_list,w_list, reward_n,done_n

# def get_state(clusteded_dic,state,i):
#     i = '%d'%(i)
#     device_idx = np.array(clusteded_dic[i])
#     return torch.cat(((state[device_idx],state[-1].view(1,-1))),0).view(-1).cpu().detach().numpy()