import gym
import matplotlib.pyplot as plt
import io
import torch 
import torch.nn as nn
import torch.optim as optim
from scipy.special import lambertw
from torch.utils.data import DataLoader, Dataset
# from collections import namedtuple, deque, defaultdict
# from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from opti_FL_compare import optimization_bf
from models.Nets import MLP, CNNMnist, CNNCifar,CNNMnist_Compare,CNNCifar_Compare,weigth_init,weigth_init_zero
from utils.options import args_parser
import numpy as np
import copy
import math
import random
import time
import json
from torch.autograd import grad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import ReplayBuffer
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from torchvision import datasets, transforms
from utils.update_device_LSTM_Compare import local_update,get_state,permute_device
from models.UpdateProb import LocalUpdate
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

args = args_parser() 
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.dataset = 'mnist'

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
 
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def add_device(idx_list,remain_index,T_sum):
    min_T = np.inf
    min_idx = np.nan
    for i in remain_index:
        temp_index = idx_list + [i]
        
        E,T,_, _, b_list, g_list = optimization_bf(f_list, p, g, C, D, 1, 2488888, B,temp_index)
#         print(b_list)
        if T < min_T:
            min_idx = i
            min_T = T
    idx_list.append(min_idx)   
#     print(b_list)
    remain_index = list(set(remain_index)- set(idx_list))
    return idx_list,remain_index,T,b_list

def B_equation(n,length,eta, beta, delta,D):
    D_sum = sum(D)
    ji = (pow((eta*beta + 1),5) -   1)*delta/beta
    result = 0
    for i in range(n):
        for j in range(n):
            result += beta * D[i]**2 *D[j]**2 *  (ji[i]**2 + ji[j]**2)
    result = result / (2 * n * (n-1)*D_sum**2 * D[0]**2)
    return result * (n - length)/length

def h(eta, beta, delta):
    return (pow((eta*beta + 1),5) -   1)*delta/beta -eta * 5 * delta

def begin_select(phi,rou,beta,eta,delta,remain_index,idx_list,T_sum):
    idx_list,remain_index,T,_ = add_device(idx_list,remain_index,T_sum)
    K_bar = np.floor(T_sum / T)
    C_bar = (1 + np.sqrt(1+4*eta * phi * K_bar**2 * 5 * (rou * h(eta, beta,  np.mean(delta)) + B_equation(n,len(idx_list),eta, beta, delta,D))))/(2 * eta
          *phi * K_bar * 5) + rou * h(eta, beta,  np.mean(delta))  + B_equation(n,len(idx_list),eta, beta, delta,D)
    while len(list(set(all_index)&(set(idx_list)))) < n:
        idx_list,remain_index,T,b_list = add_device(idx_list,remain_index,T_sum)
        K_bar = np.floor(T_sum / T)
        C_bar_new = (1 + np.sqrt(1+4*eta * phi * K_bar**2 * 5 * (rou * h(eta, beta, np.mean(delta)) + B_equation(n,len(idx_list),eta, beta, delta,D))))/(2 * eta
              *phi * K_bar * 5) + rou * h(eta, beta, np.mean(delta))  + B_equation(n,len(idx_list),eta, beta, delta,D)
        if C_bar_new > C_bar:
            break
        else:
            C_bar = C_bar_new
    return idx_list,b_list

def feature_cal_norm1(wlist, wglobal):
    state_list = []
    for i in range(100):
        para_list = []
        for (_, parameters1),(_, parameters2) in zip(wlist[i].items(),wglobal.items()):
            para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 1,dim = 0).cpu().numpy().item())
        state_list.append(sum(para_list))
#     para_list = torch.cat(para_list)
    return np.array(state_list)
def get_action(clusteded_dic,state,i,num):
    i = '%d'%(i)
    device_idx = np.array(clusteded_dic[i])
    state_cluster =  state[device_idx]
    if num <= len(device_idx):
        action = np.argsort(state_cluster)[-num:]
        return device_idx[action]
    else:
        return device_idx

def generate_shadow_fading(mean,sigma,size):
    sigma=pow(10,sigma/10)
    mean=pow(10,mean/10)
    m = np.log(pow(mean,2)/np.sqrt(pow(sigma,2)+pow(mean,2)))
    sigma= np.sqrt(np.log(pow(sigma,2)/pow(mean,2)+1))
    np.random.seed(0)
    Lognormal_fade=np.random.lognormal(m,sigma,size)
    return Lognormal_fade


def func_p(x):
    return sum(0.01 * np.array(gtk) * np.sqrt(rou / ((1-rou) * Tk + x))) - 1

def binary(func,convergence, left, right,index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  
    cur_root = left
    count = 1
    while error > convergence:
        if abs(func(left)) < convergence:
            return left
        elif abs(func(right)) < convergence:
            return right
        else:
#             print(str(count) + ' root = ' +str(cur_root))
            middle = (left + right) / 2
            if (func(left) * func(middle)) < 0:
                right = middle
            else:
                left = middle
            cur_root = left
        error = abs(func(cur_root))
        count += 1
        if count > 100:
            #print('There is no root!')
            return cur_root
    return cur_root

def generate_shadow_fading(mean,sigma,size):
    sigma=pow(10,sigma/10)
    mean=pow(10,mean/10)
    m = np.log(pow(mean,2)/np.sqrt(pow(sigma,2)+pow(mean,2)))
    sigma= np.sqrt(np.log(pow(sigma,2)/pow(mean,2)+1))
    np.random.seed(0)
    Lognormal_fade=np.random.lognormal(m,sigma,size)
    return Lognormal_fade

all_clients = 100
n = all_clients
all_index = np.arange(n).tolist()
np.random.seed(0)
random_num = np.random.random(all_clients)
dt = (10+1990*random_num)*0.001

plm = 128.1 + 37.6*np.log10(dt)
rd_number = generate_shadow_fading(0,8,n)
PathLoss_User_BS=plm+rd_number
g = pow(10,(-PathLoss_User_BS/10))
N0 =-174
N0 =pow(10,(N0/10))/1e3
i = np.array([i for i in range(0,all_clients)])
C = np.random.random(n) * 9e4 + 1e4
B = 1*1e6
alpha = 2e-28 
f_list = np.random.uniform(0.1e9,2e9,100)
p = np.random.uniform(0.001,0.19952,100)

if args.dataset == 'mnist':
    s = 2488888
    D = np.array([60000/all_clients] * all_clients)
elif args.dataset == 'fashion':
    s = 438888
else:
    s = 4894444
    D = np.array([50000/all_clients] * all_clients)

J = 1 * 5 * C * D
H = 1 * s
F = 1 * s * p
G = p * g / N0
A = 1 * 5 * alpha * D * C /2    

fenmu = B * np.log2(1+G/B)
R = np.log2(1+G/B)
Tk =  s/fenmu

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.dataset = 'mnist'
# load dataset and split users

args.iid = False

if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users= mnist_noniid(dataset_train, args.num_users,0.8)


elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users= cifar_noniid(dataset_train, args.num_users,0.5)



if args.model == 'cnn' and args.dataset == 'cifar':
    net_glob_init = CNNCifar(args=args).to(args.device)#CNNCifar_Compare().to(args.device)
elif args.model == 'cnn' and args.dataset == 'mnist':
    net_glob_init = CNNMnist_Compare().to(args.device)
elif args.model == 'cnn' and args.dataset == 'fashion':
    net_glob_init = CNNFashion().to(args.device)
elif args.model == 'mlp':
    len_in = 1
    for x in img_size:
        len_in *= x
    net_glob_init = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
else:
    exit('Error: unrecognized model')
net_glob_init.apply(weigth_init)
checkpoint = torch.load('test_mnist.pkl', map_location='cpu')

epoch = 1
net_glob_init.load_state_dict(checkpoint,False)
print(net_glob_init)
net_glob_init.train()
w_glob_init = net_glob_init.state_dict()

net_glob_zero = CNNCifar(args=args).to(args.device)
net_glob_zero.apply(weigth_init_zero)
w_init_zero = net_glob_zero.state_dict()
w_list_init = [w_init_zero] * n 
# with open('data_iid.json', 'w') as f:
#     json.dump(dict_users, f)

bias = 'iid'
if bias == 'noniid':
    preset_accuracy = torch.tensor([99.0])
    with open('data_mnist_05.json', 'r') as f:
        dict_users = json.load(f)
else:
    preset_accuracy = torch.tensor([99.0])
    with open('data_mnist_iid.json', 'r') as f:
        dict_users = json.load(f)

backup = [[34, 75, 16, 43],
[34, 75, 16, 43, 99, 53, 15, 77, 14, 87, 97, 67, 79, 26, 82, 47, 61, 69, 55, 60, 95, 46, 92, 63, 64, 54, 30, 57, 80, 48, 24, 59, 76, 40, 49, 90, 65, 58, 9, 71, 94, 81, 41, 78, 22, 29, 96, 35, 32, 25, 11, 56, 62, 2, 51, 37, 45, 6, 44, 42, 84, 28, 73, 5, 4, 39, 50, 12, 10, 1, 31, 7, 18, 33, 85, 20, 83, 52, 0, 36, 91, 98, 23],
[34, 75, 16, 43, 99, 53, 15, 77, 14, 87, 97, 67, 79, 26, 82, 47, 61, 69, 55, 60, 95, 46, 92, 63, 64, 54, 30, 57, 80, 48, 24, 59, 76, 40, 49, 90, 65, 58, 9, 71, 94, 81, 41, 78, 22, 29, 96, 35, 32, 25, 11, 56, 62, 2, 51, 37, 45, 6, 44, 42, 84, 28, 73, 5, 4, 39, 50, 12, 10, 1, 31, 7, 18, 33, 85, 20, 83, 52, 0, 36, 91, 98, 23, 38, 88, 68, 13, 3, 17, 72, 89, 21, 86, 66, 19, 8, 74, 27, 93, 70]
]


num_devices = 10
name1 = 'TrainingRecords_more_'+str(num_devices)+bias+'_'
name2 = 'Device_list_more_'+bias
name3 = 'Rewards_more_'+bias

for ep_iter in range(1):
    score = 0
    policy_loss_total = 0
    w_list = []
    episode_reward =0
    policy_reward = []
    length_temp = 0
    epoch = 0

    
    print('begin distance') 
    print('begin distance') 
    print('begin distance')         
    done = 0
         
    for roll_num in range(30):
        rou = [1]*n
        beta = [1]*n
        phi = 0.1
        eta = 0.05
        delta = np.array([1]*n)

        T_sum = 1000


        policy_reward = []
        T = 0
        E = 0
        epoch = 0
        episode_reward =0
        net_glob = copy.deepcopy(net_glob_init)
        idxs_users = [i for i in range(args.num_users)]
        w_list = copy.deepcopy(w_list_init)
        w_glob = copy.deepcopy(net_glob_init.state_dict())
        acc_list = []
        epoch = 0
        done = 0
        gtk = []
        record_list = []
        print('roll out')        
        while not done: 
            remain_index = copy.deepcopy(all_index)
            idx_list = []
            if epoch == 0:
                idxs_users = backup[0]
            elif epoch ==1:
                idxs_users = backup[1]
            else:
                idxs_users = backup[2]

            
            start =time.perf_counter()
            idxs_users,b_list = begin_select(phi,np.mean(rou),np.mean(beta),eta,delta,remain_index,idx_list,T_sum)

            #fenmu = b_list * np.log2(1+G[idxs_users]/b_list)
#             print(np.mean(rou))
#             print(np.mean(beta))
#             print(np.mean(delta))
            E,T,_, _, b_list, g_list = optimization_bf(f_list, p, g, C, D, 1, 4894444, B,idxs_users)
            end = time.perf_counter()
            print(idxs_users)
            print(len(idxs_users))
            #E = sum(A[idxs_users]*f_list[idxs_users]*f_list[idxs_users] + F[idxs_users]/(b_list*np.log2(1+G[idxs_users]/b_list)))
            #T = max(H/((b_list*np.log2(1+G[idxs_users]/b_list)))+J[idxs_users]/f_list[idxs_users])
            episode_reward += -(E + T)

            file_handle=open('Time'+name1+'more_mnist_'+str(roll_num)+'.txt',mode='a')
            file_handle.write(str(end-start))
            file_handle.write('\n')
            file_handle.close()


            #file_handle=open('ET'+name1+'more_mnist_'+str(roll_num)+'.txt',mode='a')
            #file_handle.write(str(E)+' ' + str(T))
            #file_handle.write('\n')
            #file_handle.close()

            #file_handle=open('Device'+name1+'more_mnist_'+str(roll_num)+'.txt',mode='a')
            #file_handle.write(str(idx_list))
            #file_handle.write('\n')
            #file_handle.close()
           
            
            next_state_matrix,net_glob,acc_list, w_list,done_n,rou,beta,delta = local_update(args,dataset_train,dataset_test,
    dict_users,idxs_users,epoch,net_glob,rou= rou,beta=beta,delta = delta,acc_list = acc_list, w_list = w_list, name =name1+'mnist_'+str(roll_num)+'.txt',preset = preset_accuracy)            
            
            
            w_glob = net_glob.state_dict()
            

            if epoch >= 80:
                done_n = [1] * 80
                done = 1

            epoch += 1
            done = done_n[0]
        file = open(name3+'_mnist_'+bias+'.txt',mode='a')
        file.write(str(episode_reward)+'\n')
        file.close()