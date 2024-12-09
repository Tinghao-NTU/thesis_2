# import cvxpy as cp

import numpy as np
import random
import math
import copy
from scipy.optimize import rosen, rosen_der
import time
np.set_printoptions(suppress=False)

import torch
print(torch.__version__) 



def func_fn(x,index,T,p,G,E,F):
    return pow(x,3) + (T*p[index]/(G[index])-E[index]/G[index])*x - F[index]*p[index]/(G[index])
def func_bn(x,index,result,G):
    return x*np.log2(1+G[index]/x)-result[index]

def func_bn_E(x,index):
    return G[index] * f[index] * f[index] + H[index]/(x*np.log2(1+A[index]/x)) - E[index]
def binary_BF(func,result,convergence, left, right,G, index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  
    cur_root = left
    count = 1
    if result[index] < 0:
        return 100
    else:
        while error > convergence:
            if abs(func(left,index,result,G)) < convergence:
                return left
            elif abs(func(right,index,result,G)) < convergence:
                return right
            else:
    #             print(str(count) + ' root = ' +str(cur_root))
                middle = (left + right) / 2
                if (func(left,index,result,G) * func(middle,index,result,G)) < 0:
                    right = middle
                else:
                    left = middle
                cur_root = left
            error = abs(func(cur_root,index,result,G))
            count += 1
            if count > 50:
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

def binary(func,convergence, left, right,T,p,G,E,F,index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  # 循环开始条件
    cur_root = left
    count = 1
    while error > convergence:
        if abs(func(left,index,T,p,G,E,F)) < convergence:
            return left
        elif abs(func(right,index,T,p,G,E,F)) < convergence:
            return right
        else:
#             print(str(count) + ' root = ' +str(cur_root))
            middle = (left + right) / 2
            if (func(left,index,T,p,G,E,F) * func(middle,index,T,p,G,E,F)) < 0:
                right = middle
            else:
                left = middle
            cur_root = left
        error = abs(func(cur_root,index,T,p,G,E,F))
        count += 1
        if count > 50:
            #print('There is no root!')
            return cur_root
    return cur_root

def func1(x,index):
    return s*x/(np.log2(1+g[index]*x/(N0 * b[index]))*b[index])-E[index] + G[index] * f[index] * f[index]

def optimization_bf(f, p, g, C, D, k, s, B,idx):
    E_list = []
    T_list = []
    N0 =-174
    N0 =pow(10,(N0/10))/1e3    
    num_clients = len(idx)
    print(num_clients)

    iter_num = 0
    I0 = 1
    Ik = 5
    alpha = 2e-28 
    result_list = [np.inf]
    f = f[idx]
    p = p[idx]
    g = g[idx]
    C = C[idx]
    D = D[idx]
    J = I0 * Ik * C * D
    H = I0 * s
    F = I0 * s * p
    G = p * g / N0
    A = I0 * Ik * alpha * D * C /2    

    
    T_high = 1000
    T_min = 0.01
    
    T = copy.deepcopy(T_high)
    count = 0
    while True:
    #         print(T)
        b_list = []
        for i in range(num_clients):
            b_list.append(binary_BF(func_bn,H/(T-J/(f)),1e-6, 1e-6, 5 * B / num_clients, G,index = i))
        b_sum = sum(b_list)
        ratio = b_sum/B
    
        if 0.98<= ratio and ratio <= 1:
            E = sum(A*f*f + F/(b_list*np.log2(1+G/b_list)))
            real_T = max(H/((b_list*np.log2(1+G/b_list)))+J/f)
            print(ratio)
            return E, real_T, f, p, np.array(b_list), g
        elif 0 < ratio and ratio < 0.98:
                T_high = T
                T = (T+T_min)/2
        else:
                T_min = T
                T = (T+T_high)/2       
        count += 1
        if count >= 30:
            E = sum(A*f*f + F/(b_list*np.log2(1+G/b_list)))
            real_T = max(H/((b_list*np.log2(1+G/b_list)))+J/f)
            return E, real_T, f, p, np.array(b_list), g    
