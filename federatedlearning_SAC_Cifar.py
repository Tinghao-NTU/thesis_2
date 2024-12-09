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

args.iid = True
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
bias = 'iid'


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
with open('data_iid.json', 'w') as f:
    json.dump(dict_users, f)


if bias == 'noniid':
    with open('device_index_05.json', 'r') as f:
        clusteded_dic = json.load(f)
    with open('data_05.json', 'r') as f:
        dict_users = json.load(f)
    preset_accuracy = torch.tensor([55.8])
else:
    with open('device_index_05.json', 'r') as f:
        clusteded_dic = json.load(f)
    with open('data_iid.json', 'r') as f:
        dict_users = json.load(f)
    preset_accuracy = torch.tensor([57.5])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
        
        
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.Q1  = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )        
        self.Q2  = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )           
        
        self.Q1.apply(weigth_init)
        self.Q2.apply(weigth_init)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1,q2
        

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-5, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 2 * num_actions)
        
        self.linear1.apply(weigth_init)
        self.linear2.apply(weigth_init)
        self.linear3.apply(weigth_init)

        
    def forward(self, state):
        x = F.relu(self.linear2(F.relu(self.linear1(state))))
        mu, log_std = self.linear3(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist
    
    def act(self, state, num_devices,sample=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.forward(state)
        action = dist.sample() if sample else dist.mean
#         action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return action,torch.sort(action[0])[1][-num_devices:].detach().cpu().numpy().tolist()

def soft_q_update(batch_size, 
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
           learnable_temperature = True
          ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    alpha = log_alpha.exp()
    target_entropy = -action_dim
    
    dist = policy_net(next_state)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = target_value_net(next_state, next_action)
    target_V = torch.min(target_Q1,
                         target_Q2) - alpha.detach() * log_prob
    target_Q = reward + ((1-done) * 0.99 * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = value_net(state, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)


    # Optimize the critic
    value_optimizer.zero_grad()
    critic_loss.backward()
    value_optimizer.step()    
    
    
    dist = policy_net(state)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_Q1, actor_Q2 = value_net(state, action)

    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (alpha.detach() * log_prob - actor_Q).mean()


    # optimize the actor
    policy_optimizer.zero_grad()
    actor_loss.backward()
    policy_optimizer.step()



    if learnable_temperature:
        log_alpha_optimizer.zero_grad()
        alpha_loss = (alpha *
                      (-log_prob - target_entropy).detach()).mean()
        alpha_loss.backward()
        log_alpha_optimizer.step()
          
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
        
        
action_dim = 100
state_dim  = 101
hidden_dim = 256

value_net        = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    
log_alpha = torch.zeros(1, requires_grad=True, device=device)

    
    
    
value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)


num_devices = 10
batch_size = 256
name1 = 'TrainingRecords_'+str(num_devices)+bias+'_'
name2 = 'Device_list_sac_'+bias
name3 = 'Rewards_sac_'+bias

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
CUDA_DEVICES = 0, 1, 2

net_glob_init = torch.nn.DataParallel(net_glob_init, device_ids=CUDA_DEVICES)
epoch = 0
net_glob_init .load_state_dict(checkpoint,False)
print(net_glob_init)
net_glob_init.train()
w_glob_init = net_glob_init.state_dict()
idxs_users = [i for i in range(args.num_users)]
state_matrix_init,net_glob_init,acc_list_init,_,w_list_init,count_list_init= local_update(args,dataset_train,
    dataset_test,dict_users,idxs_users,epoch,net_glob_init,acc_list = None,preset = preset_accuracy,count_list =None)

for ep_iter in range(10000):
    w_list = []
    losses = []
    all_rewards = []
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

        actions,idxs_users = policy_net.act(state_matrix,num_devices,True)

        file = open(name2+'_cifar_'+str(ep_iter)+'.txt',mode='a')
        file.write(str(np.sort(idxs_users))+'\n')
        file.close()

        next_state_matrix,net_glob,acc_list,cluster_acc,w_list, reward_n,done_n= local_update(args,dataset_train,dataset_test,dict_users,
                 idxs_users, epoch,net_glob,state_matrix = state_matrix,acc_list = acc_list, cluster_acc = cluster_acc, w_list = w_list, name =name1+'sac_cifar_'+str(ep_iter)+'.txt',preset  =preset_accuracy,count_list = count_list)
        episode_reward += reward_n
        done = done_n[0]
        if epoch > 150:
            done = 1
            done_n = [1] * 10
        if done:
            all_rewards.append(episode_reward)
            done = done_n[0]
        replay_buffer.push(state_matrix, actions.cpu().detach().numpy().reshape(-1), reward_n, next_state_matrix, done)
        if len(replay_buffer) > 1024:
            soft_q_update(batch_size)
        epoch += 1
        state_matrix = next_state_matrix        
    file = open(name3+'_cifar_'+bias+'.txt',mode='a')
    file.write(str(episode_reward)+'\n')
    file.close()
    if len(replay_buffer) > 1024:
        if np.mean(all_rewards[-5:]) > np.mean(all_rewards[-6:-1]):
            torch.save(policy_net.state_dict(), 'SAC_cifar_'+bias+'.pkl')