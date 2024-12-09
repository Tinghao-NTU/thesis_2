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
from models.Nets import MLP, CNNMnist, CNNCifar,CNNMnist_Compare,CNNCifar_Compare,weigth_init,CNNFashion,weigth_init_zero
from utils.options import args_parser
import numpy as np
import copy
import math
import random
import json
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.sampling_trans import fashion_noniid,mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from torchvision import datasets, transforms
from utils.update_device_dis_test_trans import local_update,get_state,permute_device,feature_cal_all
from models.Update import LocalUpdate


torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

args = args_parser() 
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.dataset = 'cifar'
bias = 'noniid'




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

num_frames = 500
gamma      = 0.99

all_rewards = [-10000000]
episode_reward = 0
eps = np.finfo(np.float32).eps.item()



preset_accuracy = torch.tensor([54.5])
checkpoint_DRL = torch.load('ppo_LSTM_mnist_actoriid.pkl', map_location='cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.gru_critic = nn.GRU(num_inputs, hidden_size, 1, batch_first = True)
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
        )
        self.critic2 = nn.Linear(100, 1)
        self.apply(init_weights)

    def forward(self, x):
        batch_size, seq_len, input, = x.size()
        input_critic,_ = self.gru_critic(x)
        input_critic = self.critic1(input_critic).view(batch_size,seq_len)
        value = self.critic2(input_critic)
        return value

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size     
        self.gru_actor = nn.GRU(num_inputs, hidden_size, 1, batch_first = True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        batch_size, seq_len, input, = x.size()
        input_actor,_ = self.gru_actor(x)
        mu    = self.actor(input_actor).view(batch_size,seq_len)     
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist



def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.array_split(ids,batch_size //mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist = model_actor(state)
            value = model_critic(state)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            print('loss is ' + str(loss))
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()


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
lr               = 1e-4
num_steps        = 200
mini_batch_size  = 30
ppo_epochs       = 10


model_actor = Actor(5, 100, 256).to(device)
model_critic = Critic(5, 100, 256).to(device)

model_actor.load_state_dict(checkpoint_DRL,False)

optimizer_actor = optim.Adam(model_actor.parameters(), lr=lr)
optimizer_critic = optim.Adam(model_critic.parameters(), lr=lr)


def generate_shadow_fading(mean,sigma,size):
    sigma=pow(10,sigma/10)
    mean=pow(10,mean/10)
    m = np.log(pow(mean,2)/np.sqrt(pow(sigma,2)+pow(mean,2)))
    sigma= np.sqrt(np.log(pow(sigma,2)/pow(mean,2)+1))
    np.random.seed(0)
    Lognormal_fade=np.random.lognormal(m,sigma,size)
    return Lognormal_fade




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

epoch = 1
#net_glob_init.load_state_dict(checkpoint,False)
print(net_glob_init)
net_glob_init.train()
w_glob_init = net_glob_init.state_dict()

all_clients = 100
n = all_clients


net_glob_zero = CNNCifar(args=args).to(args.device)
net_glob_zero.apply(weigth_init_zero)
w_init_zero = net_glob_zero.state_dict()
w_list_init = [w_init_zero] * n 

para_list = []
for (_, parameters1) in w_glob_init.items():
    para_list.append((parameters1.view(-1)- 0).cpu())
inti_s = torch.norm(torch.cat(para_list), p = 2,dim = 0).item()

num_devices = 10
log_probs = []
values    = []
states    = []
actions   = []
rewards   = []
masks     = []
entropy = 0
all_rewards = []
max_reward = -100000

name1 = 'TrainingRecords_LSTM_changethird_test_trans_'+str(num_devices)+bias+'_'
name2 = 'Device_list_ppo_LSTM_changethird_test_trans_'+bias
name3 = 'Rewards_ppo_LSTM_changethird_test_trans_'+bias

update_counts = 0
for ep_iter in range(2,30):

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


    w_list = []
    losses = []

    all_clients = 100
    n = all_clients
    seed = ep_iter
    np.random.seed(seed)
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
    f = np.random.uniform(0.1e9,2e9,100)
    p = np.random.uniform(0.001,0.19952,100)

    if args.dataset == 'mnist':
        s = 2488888
        D = np.array([60000/all_clients] * all_clients)
    if args.dataset == 'fashion':
        s = 438888
    else:
        s = 4894444
        D = np.array([50000/all_clients] * all_clients)

    dict_users = cifar_noniid(dataset_train, args.num_users,0.5,seed)
    length_temp = 0

    episode_reward =0
    
    done = 0
    epoch = 1
    net_glob = copy.deepcopy(net_glob_init)
    count_list = np.zeros(100)
    w_list = copy.deepcopy(w_list_init)
    w_glob = copy.deepcopy(net_glob_init.state_dict())
    cluster_acc = None
    acc_list = []
    while done != 1:
        if epoch == 1:
            state_matrix = np.concatenate((feature_cal_all(w_list,w_glob).reshape(-1,1),f.reshape(-1,1)/1e9, p.reshape(-1,1),dt.reshape(-1,1), D.reshape(-1,1)/1000),1) 
        
        update_counts += 1
        
        state = torch.FloatTensor(state_matrix).unsqueeze(0).to(device)
        print(state)
        dist = model_actor(state)
        value = model_critic(state)

        action = dist.sample()
        pmt = np.argsort(count_list)
        idxs_users = pmt[torch.multinomial(F.softmax(action,dim = 1), 10, replacement=False).detach().cpu().view(-1)]
        #torch.sort(F.softmax(action,dim = 1).view(-1))[1][-10:].cpu().tolist()
        

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        log_probs.append(log_prob)
        values.append(value)
   
        
        
#         print(np.sort(idxs_users))
        if epoch % 10 == 0:
            print(count_list)
            print(action)
        
        next_state_matrix,net_glob,acc_list,cluster_acc,w_list, reward_n,done_n= local_update(args,dataset_train,dataset_test,dict_users,
                    idxs_users, epoch,net_glob, acc_list = acc_list, cluster_acc = cluster_acc, w_list = w_list, name =name1+'ppo_cifar_'+str(ep_iter)+'.txt',preset  =preset_accuracy,count_list = count_list,seed = seed)

        rewards.append(torch.FloatTensor(np.array([reward_n])).unsqueeze(1).to(device))
        masks.append(torch.tensor([1 - done_n[0]]).unsqueeze(1).to(device))
        states.append(state)
        actions.append(action)
        
        episode_reward += reward_n
        done = done_n[0]
        if epoch > 80:
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
