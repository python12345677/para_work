import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.linalg import eigh

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
    for i in range(Na):
        for j in range(Na):
            if i!=j:
                if (Ha1[j,j]-Ha1[i,i])>=0:
                    tmp_sp = basis(Na,j)*basis(Na,i).dag()
                    tmp_n_th = abs(1/(np.exp(Ha1[j,j]-Ha1[i,i])-1))
                    c_ops.append(np.sqrt(tmp_n_th+1)*tmp_sp.dag())
                    c_ops.append(np.sqrt(tmp_n_th)*tmp_sp)

def LZS_Ham(Ad, detuning, tunnel, vd, maxm):
    ## 实验参数区 单位：GHz
    h=1;#Planck constant 
    gamma=0.1;#decoherence rate of qubit
    vr=5.198;#resonator frequency
    vprobe=vr;#probe frequency
    alpha=0.001;#dimensionless dissipation strength
    T = 0.1
    g=0.081;#couple strength between qubit and cavity
    kbT = 86.17333262145/4.1356676969 * T;#GHz #k_b*10mK #86.17333262145 ueV h = 4.1356676969×10-15 eV·s
    ## Floquet计算参数区
    dima = 2;# atom系统的维度=2(固定，更改需一并修改H0定义)
    maxm = maxm;# floquet张开的最大光子数值
    pi = np.pi
    ## 额外参数区
    kappa = 0.01197
    kappaext = 0.007
    kappaint = kappa - kappaext  
    H0 = tunnel/2 * sigmax() + detuning / 2 * sigmaz()
    N = 2 * maxm + 1
    idf = identity(N)
    mhat = tensor(qdiags(range(-maxm,maxm+1),0),identity(dima))
    mhatin = Qobj(np.diag(np.ones(2*maxm,dtype=np.float64),-1))
    mhatde = Qobj(np.diag(np.ones(2*maxm,dtype=np.float64),1))
    mhatin = tensor(mhatin, identity(dima))
    mhatde = tensor(mhatde, identity(dima))
    H0 = tensor(idf,H0)
    sigmazt = tensor(identity(N),sigmaz())
    sigmaxt = tensor(identity(N),sigmax())
    Hf = H0 + h * vd * mhat - Ad / 4 * sigmazt * (mhatin + mhatde)
    return Hf





