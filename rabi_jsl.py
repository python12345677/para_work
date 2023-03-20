import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
from scipy.signal import convolve

import matplotlib.axes._axes as axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

def gaussian_quantum(x1,sigma1):
    step1 = x1[1]-x1[0]
    n1 = len(x1)
    x_truc = int(np.ceil(3 * sigma1/step1))
    print(step1,x_truc)
    kernel = np.zeros(2*x_truc+1)
    for x in range(-x_truc, x_truc + 1):
        # 二维高斯函数
        v = 1.0 / (2 * np.pi * sigma1) * np.exp(-1.0 / (2 * sigma1 ** 2) * ((x*step1) ** 2))
        kernel[x + x_truc] = v  # 高斯函数的x和y值 vs 高斯核的下标值
    kernel = kernel / np.sum(kernel)
    return kernel

def add_heat(Ha1:Qobj,Na:int,c_ops:list):
    """Fetches rows from a Smalltable.

    add heat to make pn = boltzmann distribution

    Args:
        Ha1: diag matrix for qubit system
        Na: the dim for qubit
        c_ops: the collapse operator

    Returns:
        None

    Raises:
        None
    """
    kbT = 1.5
    gamma = 0.01
    for i in range(Na):
        for j in range(Na):
            if i!=j:
                if (Ha1[j,j]-Ha1[i,i])>=0:
                    tmp_sp = basis(Na,j)*basis(Na,i).dag()
                    tmp_n_th = abs(1/(np.exp((Ha1[j,j]-Ha1[i,i])/kbT)-1))
                    c_ops.append(np.sqrt((tmp_n_th+1)*gamma)*tmp_sp.dag())
                    c_ops.append(np.sqrt(tmp_n_th*gamma)*tmp_sp)

def py_coeff1(t, args):
    wr = args['wr']
    return np.exp(1j *wr*t)
def py_coeff2(t, args):
    wr = args['wr']
    return np.exp(-1j *wr*t)

def CGplot(ax: axes.Axes, data_x: np.ndarray, data_y: np.ndarray, data_z: np.ndarray, title=None, namex=None,
           namey=None, namez=None):
    if title == None:
        title = ''
    if namex == None:
        namex = ''
    if namey == None:
        namey = ''
    if namez == None:
        namez = ''

    # color map
    r = np.array([0.2, 0, 0])
    red = np.array([1, 0, 0])
    white = np.array([1, 1, 1])
    blue = np.array([0, 0, 1])
    b = np.array([0, 0, 0.2])
    color_num = 100
    level_1 = 0.2
    level_2 = 0.5
    level_3 = 0.8
    tmp_range1 = range(0, int(color_num * level_1))
    tmp_range2 = range(int(color_num * level_1), int(color_num * level_2))
    tmp_range3 = range(int(color_num * level_2), int(color_num * level_3))
    tmp_range4 = range(int(color_num * level_3), color_num)
    cdefine = np.zeros((color_num, 3), dtype=np.float64)
    for i in tmp_range1:
        cdefine[i, :] = (b * (color_num * level_1 - i + 1) + blue * (i - 1)) / (color_num * level_1)
    for i in tmp_range2:
        cdefine[i, :] = (blue * (color_num * level_2 - i + 1) + white * (i - color_num * level_1 - 1)) / (
                    color_num * (level_2 - level_1))
    for i in tmp_range3:
        cdefine[i, :] = (white * (color_num * level_3 - i + 1) + red * (i - color_num * level_2 - 1)) / (
                    color_num * (level_3 - level_2))
    for i in tmp_range4:
        cdefine[i, :] = (red * (color_num - i + 1) + r * (i - color_num * level_3 - 1)) / (
                    color_num * (1 - level_3) * 1)
    cdefine[cdefine > 1] = 1
    cdefine[cdefine < 0] = 0

    zmax = abs(data_z.max())
    zmin = abs(data_z.min())
    # color map and normalize
    cg_cmp = ListedColormap(cdefine, name='cg_cmp')
    norm = Normalize(zmin, zmax)

    # plot
    color_bar = plt.colorbar(ax=ax, mappable=ScalarMappable(cmap=cg_cmp, norm=norm), label='sss')
    Zlevels = np.linspace(zmin, zmax, 100)
    ax.contourf(data_x, data_y, data_z, levels=Zlevels, cmap=cg_cmp, norm=norm)

    # label
    color_bar.set_label(title, rotation=270, size=20, loc='center')
    ax.set_xlabel(namex)
    ax.set_ylabel(namey)
    ax.set_title(namez)



def one_curve_test(probe_lis,Hc=None,g=None,eps=None,t2=None):
    # cavity
    Nc = 10
    Hc = Hc
    kappa = 0.00633
    # wp = wp
    # atom
    Na = 2
    eps = eps
    t2 = t2
    # couple
    g = g
    # operator
    a = tensor(qeye(Na), destroy(Nc))
    ad = a.dag()
    sz = tensor(sigmaz(), qeye(Nc))
    sx = tensor(sigmax(), qeye(Nc))
    sm = tensor(sigmaz(), qeye(Nc))
    # collapse operator
    c_ops= []
    # c_ops.append(np.sqrt(0.1) * sm)
    n_th = 0.005
    c_ops.append(np.sqrt((1+n_th)*kappa) * a)
    c_ops.append(np.sqrt((n_th)*kappa) * ad)
    # Hamiltonian
    Hc = Hc*ad*a
    # tranformation for qubit
    Ha0 = eps/2*sigmaz() + t2/2*sigmax()
    w,v = eigh(Ha0)
    Ha1 = np.transpose(np.conj(v))@np.array(Ha0)@v
    Ha1 = Qobj(Ha1)
    tmp_sm = sigmaz()
    # if eps<0:
    #     tmp_sm = sigmap()
    # else:
    #     tmp_sm = sigmam()
    sigmam1 = np.transpose(np.conj(v))@np.array(tmp_sm)@v
    sigmam1 = Qobj(sigmam1)
    sm1 = tensor(sigmam1,qeye(Nc))
    Hi1 = 1*g*(ad+a)*sm1
    c_ops2 = []
    add_heat(Ha1,Na,c_ops2)
    for each in c_ops2:
        c_ops.append(tensor(each, qeye(Nc)))
    Ha1 = tensor(Ha1,qeye(Nc))
    # all Hamiltonian
    H0 = Hc + Ha1 + Hi1
    # probe
    Hp1 =  [np.sqrt(kappa*0.3*0.0001)*(ad), py_coeff2]
    Hp2 =  [np.sqrt(kappa*0.3*0.0001)*(a), py_coeff1]
    H = [H0,Hp1,Hp2]
    # solve
    psi0 = tensor(basis(Na,0),fock(Nc,0))
    time = np.linspace(0,10000,10000)
    # res = mesolve(H,psi0,time,c_ops,[a,ad*a,tensor(qdiags([1,0],0),qeye(Nc))],args={'wr': wp})
    # aexpt = abs(res.expect[0][-int(len(time)/3):-1])
    # ret = np.sum(aexpt)/len(aexpt)
    # atom_exp = abs(res.expect[2][-1])
    # print(atom_exp)
    probe_lis = probe_lis
    spec2 = spectrum(H0,probe_lis,c_ops,ad,a)
    return spec2


N1 = 10000
N2 = 101
probe_lis = np.linspace(4.95,5.04,N1)
eps_lis = np.linspace(-18,18,N2)#6->36
res_lis = np.zeros((N1,N2))

def main():
    for i,eps in enumerate(eps_lis):
        print("eps:",eps)
        # a = parfor(one_curve_test, probe_lis, Hc=4.9913,g=0.1,eps=eps,t2=4.2)
        a = one_curve_test(probe_lis,Hc=4.993,g=0.11,eps=eps,t2=4.2)
        # print(a)
        res_lis[:,i]=np.sqrt(a)
    return a
if __name__ == '__main__':
    t1 = time.time()
    main()
    plt.figure(figsize=(14,5))
    ax = plt.subplot(121)
    CGplot(ax,eps_lis,probe_lis,res_lis)
    ax = plt.subplot(122)

    kernal = gaussian_quantum(eps_lis,1.5)
    print(kernal)
    for i in range(N1):
        tmp = convolve(res_lis[i,:],kernal,'same')
        res_lis[i,:] = tmp
    # kernal = gaussian_quantum(probe_lis,0.001)
    # print(kernal)
    # for i in range(N2):
    #     tmp = convolve(res_lis[:,i],kernal,'same')
    #     res_lis[:,i] = tmp
    # res_lis = res_lis/np.max(res_lis)*0.4
    CGplot(ax,eps_lis,probe_lis,res_lis)
    t2 = time.time()
    print(t2-t1)
    # plt.legend()
    plt.show()
