#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 07:54:28 2020

@author: simo94
"""

" Script for importing file from external data"

import os 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


           
def conv_rate(dof,err):

    ' Compute convergence rate '   
    l = dof.shape[0]
    rate = np.zeros(l-1)
    
    for i in range(l-1):
        rate[i] = np.log(err[i]/err[i+1])/np.log(dof[i+1]/dof[i])
        
    rate = np.mean(rate)

    print('Convergence rate is ' + str(rate))

    return rate


            
def Fourier_trans(dt,y):
    
    N = len(y)
    Y  = np.fft.fft(y)
    freq = np.fft.fftfreq(N,dt)

    plt.figure()
    plt.plot((2*np.pi)*freq[1:np.int(N/2)], np.abs(Y)[1:np.int(N/2)])
    plt.xlabel('$\omega (HZ)$')
    plt.ylabel('FFT magnitude (power)')
    plt.yscale('log')
    
    idx = find_peaks(np.abs(Y)[1:np.int(N/2)])
    print('maximum angular frequency for w: ',(2*np.pi)*freq[1 + idx[0]])
            

# get home directory from python 



dt = 0.002
tf = 5.0
nt = np.int(tf/dt)

t_vec = np.arange(1,nt+2)*dt

os.getenv("HOME")

#data_path = '/home/simo94/Documents/Data_folder'
#fin_path = '/RK4/Uniform/2D/Data_CG2BDM1DG0'



outputfile1 = '/home/simo94/SW_MFEM/scripts/SW_RK4/Linearized/Data_CG1RT1DG0/dev_sol_N_20_CG1RT1DG0.npy'
outputfile2 = '/home/simo94/SW_MFEM/scripts/SW_RK4/Linearized/Data_CG1RT1DG0/dev_scalars_N_20_CG1RT1DG0.npy'


dev_sol = np.load(outputfile1)
dev_scalars = np.load(outputfile2)




#Compute Fourier transform of the quantities 
Fourier_trans(dt,dev_sol[:,0])
Fourier_trans(dt,dev_sol[:,1])
Fourier_trans(dt,dev_sol[:,2])


#### Deviations solution ####
        
   
#  Plot oscillatory deviations from initial condition over time 
fig = plt.figure()
plt.plot(t_vec,dev_sol[:,0])
plt.xlabel('t')
plt.ylabel('L2 error')
plt.title('u')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
fig = plt.figure()
plt.plot(t_vec,dev_sol[:,1])
plt.xlabel('t')
plt.ylabel('L2 error')
plt.title('v')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
fig = plt.figure()
plt.plot(t_vec,dev_sol[:,2])
plt.xlabel('t')
plt.ylabel('L2 error')
plt.title('h')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))


# N = np.array([20,30,40,50,60])
# w = np.array([1191.42,1788.91,2386.01,2998.72,3592.54])

# rate_w = conv_rate(N,w)
# fig, ax = plt.subplots()
# ax.plot(N,w,linestyle = '-.',marker = 'o', label = 'rate_w: ' + '%.4g' %-rate_w)
# ax.set_xlabel('N')
# ax.set_ylabel('$\omega$')
# ax.legend()


# #Rescale scalar variables and plot deviations for highest dof

scalars_norm = dev_scalars 


#for i in range(dev_scalars.shape[1]):
#    scalars_norm[:,i] = (dev_scalars[:,i] - dev_scalars[0,i])/dev_scalars[0,i]

   
#plot Scalar quantities and check for convergence
fig = plt.figure()
plt.subplot(2,2,1)
plt.plot(t_vec,scalars_norm[:,0])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Potential Energy')
plt.ylabel('$E$')
plt.xlabel('t')
plt.subplot(2,2,2)
plt.plot(t_vec,scalars_norm[:,1])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Kinetic Energy')
plt.xlabel('t')
plt.ylabel('$E$')
plt.subplot(2,2,3)
plt.plot(t_vec,scalars_norm[:,2])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Mass')
plt.ylabel('$(m - m_{0})/m_{0}$')
plt.xlabel('t')
plt.subplot(2,2,4)
plt.plot(t_vec,scalars_norm[:,3])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Total Energy')
plt.ylabel('$(E - E_{0})/E_{0}$')
plt.xlabel('t')
fig.tight_layout()
plt.savefig('scalars')





  

# # Compute convergence rate
# rate_u,rate_h = conv_rate(dof,err_sol)
   

# fig, ax = plt.subplots()
# ax.plot(np.sqrt(dof[:,0]),err_sol[:,0],linestyle = '-.',marker = 'o',label = 'u:'+ "%.4g" %rate_u)
# ax.plot(np.sqrt(dof[:,1]),err_sol[:,1],linestyle = '-.',marker = 'o',label = 'h:'+ "%.4g" %rate_h)
# ax.set_xlabel('$\sqrt{n_{dof}}$')
# ax.set_ylabel('L2 error')
# ax.set_yscale('log')
# ax.set_xscale('log')           
# ax.legend(loc = 'best')


