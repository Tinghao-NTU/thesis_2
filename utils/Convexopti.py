import numpy as np
import random
import math
import copy
from scipy.optimize import rosen, rosen_der
import time
import torch
np.set_printoptions(suppress=False)

def binary(func,convergence, left, right,index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  # 循环开始条件
    cur_root = left
    count = 1
    while error > convergence:
        if abs(func(left,index)) < convergence:
            return left
        elif abs(func(right,index)) < convergence:
            return right
        else:
#             print(str(count) + ' root = ' +str(cur_root))
            middle = (left + right) / 2
            if (func(left,index) * func(middle,index)) < 0:
                right = middle
            else:
                left = middle
            cur_root = left
        error = abs(func(cur_root,index))
        count += 1
        if count > 1000:
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
def func_fn(x,index):
    return pow(x,3) + (T*p[index]/(G[index])-E[index]/G[index])*x - F[index]*p[index]/(G[index])
def func_bn(x,index):
    return s/(x*np.log2(1+A[index]/x))+F[index]/f[index] - T#x*np.log2(1+A[index]/x) - s/(T-F[index]/f[index])
def func_bn_E(x,index):
    return G[index] * f[index] * f[index] + H[index]/(x*np.log2(1+A[index]/x)) - E[index]#x*np.log2(1+A[index]/x) - s/(T-F[index]/f[index])

def func1(x,index):
    return s*x/(np.log2(1+g[index]*x/(N0 * b[index]))*b[index])-E[index] + G[index] * f[index] * f[index]

all_clients = 100;
n = all_clients
np.random.seed(0)
random_num = np.random.random(all_clients)
dist = (50+200*random_num)*0.001 #250*1.414*np.random.random(50);# 在500*500正方形区域内
dist1 = 50*0.001
dist0 = 10*0.001

# dist = np.loadtxt('distance.txt')



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
B = 20 * 1e6

eta = 0.5
gamma = 1
l = 1
a = (np.log2(1/epsilon)*2*pow(l,2))/(gamma*gamma*xi)
v = 2/((2-l*delta)*delta*gamma)

Ik = v*np.log2(1/eta)
I0 = a/(1-eta)


def optimize(g, C):
    A = g * p / N0
    F = 5 * C * D 
    G = 5  * k * C * D 
    H = (s * p)
    count = 0 
    ratio_b  = 0
    tola = 0.1
    start = time.time()
    while not (1-tola <= ratio_b and ratio_b <=1):
        count += 1
        f_list = []
        b_list = []
        tola = 0.1
        f_max = np.sqrt((E-H/A)/G)#np.ones(num_clients) * f_max#np.sqrt((E-H/A)/G)
        f_min = F/(T-s/A)
        for i in range(n):
            f_temp = binary(func_fn,1e-3, 0.0001, 2e9,index = i)
            f_list.append(f_temp)
        f = np.array(f_list)#np.ones(n) * 2e9#
    #     print(f)
        for i in range(n):
            b_temp = binary(func_bn,1e-3, 0.01, 3 * B / n ,index = i)
            b_list.append(b_temp)
        b = np.array(b_list)
        b_sum = sum(b)
        ratio_b = b_sum/B
        if (1-tola) <= ratio_b <= 1:
            T_out = T
            break
        elif ratio_b < 1 - tola:
            T_max = T
            T = (T + T_min)/2
        elif ratio_b > 1:
            T_min = T
            T = (T_max + T)/2

    return T_list[-1]