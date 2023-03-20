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