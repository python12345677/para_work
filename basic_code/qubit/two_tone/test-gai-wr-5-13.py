# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:40:56 2023

@author: lifangge
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip.rhs_generate import rhs_clear
from qutip.solver import Options, config
from qutip.rhs_generate import rhs_clear, _td_wrap_array_str
from qutip.cy.utilities import _cython_build_cleanup
from datetime import datetime 
#from cuda_mesolve import mesolve
import math
from scipy import signal
import pickle

wq = 4.95*2*np.pi
sigma_x = qt.sigmax()
sigma_z = qt.sigmaz()
lamda_x = 0.01*2*np.pi
lamda_z = 0.024*2*np.pi
wz = 0.02*2*np.pi
num_delta = 50
num_t = 50
delta = np.linspace(-0.05, 0.05, num_delta)
T = np.linspace(1, 400, num_t)
output = np.zeros((num_delta, num_t))

# 定义迭代函数
def compute_output(index):
    i = index//num_t
    j = index % num_t
    # print(i,j)
    wx = wq + delta[i] * 2 * np.pi
    t = np.linspace(0, T[j], 500)
    H0 = 1/2 * wq * sigma_z
    Hx = lamda_x * sigma_x
    Hz = lamda_z * sigma_z
    H = [H0, [Hx, 'cos(wx*t)'], [Hz, 'cos(wz*t)']]
    e_state = qt.basis(2, 0)
    psi0 = e_state * e_state.dag()
    opts = qt.Options(rhs_reuse=True)
    result = qt.mesolve(H, psi0, t, [], qt.sigmaz(),args={'wx':wx,'wz':wz},options=opts)
    rhs_clear()
    return result.expect[0][-1]

if __name__ == '__main__':
    # 并行计算
    start_time = datetime.now()
    # for j in range(num_t):
    tmp = qt.parallel.parallel_map(compute_output, range(num_delta*num_t),progress_bar=True)
    output = np.reshape(tmp,(num_delta,num_t))
    end_time = datetime.now()
    print(output)
    # 打印计算时间
    print("计算时间:", end_time - start_time)
    # 绘制结果
    plt.contourf(T, delta,output, levels=100, cmap='jet')
    plt.colorbar()
    plt.show()
    # ...

