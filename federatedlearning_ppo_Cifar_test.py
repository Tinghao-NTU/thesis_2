import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
import os
import time
import copy
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.autograd as autograd 
import io
import torch 
import scipy.stats
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from models.Nets import MLP, CNNMnist, CNNCifar,CNNMnist_Compare,CNNCifar_Compare,weigth_init,CNNFashion
from utils.options import args_parser
import numpy as np
import copy
import math
import random
import json
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.sampling import fashion_noniid,mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from torchvision import datasets, transforms
from utils.update_device_HEFL import local_update,get_state,permute_device
from models.Update import LocalUpdate


torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

args = args_parser() 
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.dataset = 'cifar'
bias = 'noniid'


if args.dataset == 'fashion':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.FashionMNIST(
        '../data/fashion', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    dataset_test = datasets.FashionMNIST(
            '../data/fashion', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))

    dict_users  = fashion_noniid(dataset_train, args.num_users,0.8)


if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users= mnist_noniid(dataset_train, args.num_users,0.8)

elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users = cifar_noniid(dataset_train, args.num_users,0.8)
num_frames = 500
gamma      = 0.99

all_rewards = [-10000000]
episode_reward = 0
eps = np.finfo(np.float32).eps.item()


epoch = 0
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
print(net_glob_init)
net_glob_init.train()
w_glob_init = net_glob_init.state_dict() 
idxs_users = [i for i in range(args.num_users)]
# with open('data_iid.json', 'w') as f:
#     json.dump(dict_users, f)

    
if bias == 'noniid':
    with open('device_index_05.json', 'r') as f:
        clusteded_dic = json.load(f)
    with open('data_05.json', 'r') as f:
        dict_users = json.load(f)
    preset_accuracy = torch.tensor([55.5])
else:
    with open('device_index_05.json', 'r') as f:
        clusteded_dic = json.load(f)
    with open('data_iid.json', 'r') as f:
        dict_users = json.load(f)
    preset_accuracy = torch.tensor([56.8])
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1) 

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.array_split(ids,batch_size //mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :] 

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            print('loss is ' + str(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

hidden_size      = 256
lr               = 1e-5
num_steps        = 400
mini_batch_size  = 50
ppo_epochs       = 10
max_reward = -100000


saved_ppo = 'ppo_cifar_'+bias+'.pkl'

model = ActorCritic(101, 100, hidden_size).to(device)
checkpoint = torch.load(saved_ppo, map_location='cpu')
model.load_state_dict(checkpoint,False)



optimizer = optim.Adam(model.parameters(), lr=lr)

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
checkpoint = torch.load('test_cifar.pkl', map_location='cpu')
CUDA_DEVICES = 0, 1

net_glob_init = torch.nn.DataParallel(net_glob_init, device_ids=CUDA_DEVICES)
epoch = 0
net_glob_init .load_state_dict(checkpoint,False)
print(net_glob_init)
net_glob_init.train()
w_glob_init = net_glob_init.state_dict()
idxs_users = [i for i in range(args.num_users)]
state_matrix_init,net_glob_init,acc_list_init,_,w_list_init,count_list_init= local_update(args,dataset_train,
    dataset_test,dict_users,idxs_users,epoch,net_glob_init,acc_list = None,preset = preset_accuracy,count_list =None)

num_devices = 100
log_probs = []
values    = []
states    = []
actions   = []
rewards   = []
masks     = []
all_rewards = []
entropy = 0

name1 = 'TrainingRecords_ppo_'+str(num_devices)+bias+'_'
name2 = 'Device_list_ppo_'+bias
name3 = 'Rewards_ppo_'+str(num_devices)+bias

update_counts = 0
for ep_iter in range(10000):
    w_list = []
    losses = []

    length_temp = 0

    episode_reward =0
    
    done = 0
    epoch = 1
    state_matrix = copy.deepcopy(state_matrix_init)
    net_glob = copy.deepcopy(net_glob_init)
    count_list = copy.deepcopy(count_list_init)
    acc_list = copy.deepcopy(acc_list_init)
    w_list = copy.deepcopy(w_list_init)
    w_glob = net_glob.state_dict()
    cluster_acc = None
    while done != 1:
        update_counts += 1
        state = torch.FloatTensor(state_matrix).view(1,-1).to(device)
        dist, value = model(state)
        action = dist.sample()
        idxs_users = torch.multinomial(F.softmax(action,dim = 1), num_devices, replacement=False).detach().cpu().view(-1).tolist()

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()        
        log_probs.append(log_prob)
        values.append(value)

                
        next_state_matrix,net_glob,acc_list,cluster_acc,w_list, reward_n,done_n= local_update(args,dataset_train,dataset_test,dict_users,
                 idxs_users, epoch,net_glob,state_matrix = state_matrix, acc_list = acc_list, cluster_acc = cluster_acc, w_list = w_list, name =name1+'cifar_'+str(ep_iter)+'.txt',preset  =preset_accuracy,count_list = count_list)

        rewards.append(torch.FloatTensor(np.array([reward_n])).unsqueeze(1).to(device))
        masks.append(torch.tensor([1 - done_n[0]]).unsqueeze(1).to(device))
        states.append(state)
        actions.append(action)
        
        episode_reward += reward_n
        done = done_n[0]
        if epoch > 150:
            done = 1
            done_n = [1] * 10
        if done:
            all_rewards.append(episode_reward)
            done = done_n[0]

        epoch += 1
        state_matrix = next_state_matrix  
        
        
 
    file = open(name3+'_cifar_'+bias+'.txt',mode='a')
    file.write(str(episode_reward)+'\n')
    file.close()