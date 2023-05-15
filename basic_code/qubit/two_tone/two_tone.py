from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft,fftshift
from qutip.rhs_generate import rhs_clear


# TODO 根据PhysRevLett.94.123602文献进行two_tone现象的观察，没有腔，单纯的点。


sx = sigmax()
sz = sigmaz()
rho11 = (sigmaz()+1)/2
rho22 = rho11-sz
rho12 = sigmap()

A = 0.1
eps = 0
t = 5
w1 = 5
H0 = (eps+0)/2*sx + t/2*sz
H1 = [H0,[sx,'{0}*cos({1}*t)'.format(A,w1)]]


psi0 = basis(2,0)
N = 100000
tmax = 1000
Fs = int(N/tmax)
tlist = np.linspace(0,tmax,N)

res0 = mesolve(H0,psi0,tlist,[],[rho11])
plt.subplot(1,3,1)
# plt.plot(tlist,res0.expect[0])
plt.plot(tlist,np.imag(res0.expect[0]))
# plt.ylim(0,1)

# psi0 = basis(2,0)
# tlist = np.linspace(0,10,N)
res1 = mesolve(H1,psi0,tlist,[],[rho11])
plt.subplot(1,3,2)
plt.plot(tlist,res1.expect[0])
# plt.ylim(0,1)



y = res1.expect[0]
# y = np.imag(res0.expect[0])
N = len(y)
fft_y=fftshift(fft(y))                          #快速傅里叶变换
x = np.linspace(0,Fs,N)-Fs/2  
x = x*2*np.pi          # 频率个数
abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
normalization_y=abs_y/N            #归一化处理（双边频谱）dd
plt.subplot(1,3,3)
plt.plot(x,normalization_y)
plt.xlim(-1,1)
rhs_clear()
plt.show()







