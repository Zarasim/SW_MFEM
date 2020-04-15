#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:40:29 2020

@author: simo94

Plot convergence rate for Steady_State solution

"""

import numpy as np
import matplotlib.pyplot as plt
import math


RT1 = np.load('CG1RT1DG0_ref.npy')
err_RT1 = RT1[0]
dof_RT1 = np.sqrt(RT1[1])

RT2 = np.load('CG2RT2DG1_ref.npy')
err_RT2 = RT2[0]
dof_RT2 = np.sqrt(RT2[1])

BDM1 = np.load('CG1BDM1DG1_ref.npy')
err_BDM1 = BDM1[0]
dof_BDM1 = np.sqrt(BDM1[1])


BDM2 = np.load('CG2BDM2DG2_ref.npy')
err_BDM2 = BDM2[0]
dof_BDM2 = np.sqrt(BDM2[1])

                           
def conv_rate(dof,err):
    
    """
        input: Error matrix: [u, h] and [dof_u, dof_h]
        
    """
    
    l = 4
    rate_h = np.zeros(l-1)
    rate_u = np.zeros(l-1)
    
    
    for i in range(l-1):
        
        rate_u[i] = math.log(err[i,0]/err[i+1,0])/math.log(math.sqrt(dof[i+1,0]/dof[i,0]))
        rate_h[i] = math.log(err[i,1]/err[i+1,1])/math.log(math.sqrt(dof[i+1,1]/dof[i,1]))
    
    print('mean convergence rate of u is ' + str(np.mean(rate_u)))
    print('mean convergence rate of h is ' + str(np.mean(rate_h)))


conv_rate(dof_RT1,err_RT1)
conv_rate(dof_RT2,err_RT2)
conv_rate(dof_BDM1,err_BDM1)
conv_rate(dof_BDM2,err_BDM2)




fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111) 
            
ax.plot(dof_RT1[:,0],err_RT1[:,0], color='blue',marker = '^',linestyle = '-.', lw=0.5, label = 'u - RT1DG0')
ax.set_yscale('log')

ax.plot(dof_RT2[:,0],err_RT2[:,0], color='cyan',marker = '^',linestyle = '-.', lw=0.5, label = 'u - RT2DG1')
ax.set_yscale('log')

ax.plot(dof_BDM1[:,0],err_BDM1[:,0], color='green',marker = '^',linestyle = '-.', lw=0.5, label = 'u - BDM1DG1')
ax.set_yscale('log')
            
ax.plot(dof_BDM2[:,0],err_BDM2[:,0], color='red',marker = '^',linestyle = '-.', lw=0.5, label = 'u - BDM2DG2')
ax.set_yscale('log')
           
        
        
ax.plot(dof_RT1[:,1],err_RT1[:,1], color='blue',marker = 'o',linestyle = '-.', lw=0.5, label = 'h - RT1DG0')
ax.set_yscale('log')

ax.plot(dof_RT2[:,1],err_RT2[:,1], color='cyan',marker = 'o',linestyle = '-.', lw=0.5, label = 'h - RT2DG1')
ax.set_yscale('log')

ax.plot(dof_BDM1[:,1],err_BDM1[:,1], color='green',marker = 'o',linestyle = '-.', lw=0.5, label = 'h - BDM1DG1')
ax.set_yscale('log')
            
ax.plot(dof_BDM2[:,1],err_BDM2[:,1], color='red',marker = 'o',linestyle = '-.', lw=0.5, label = 'h - BDM2DG2')
ax.set_yscale('log')


#ax.set_yscale('log')
ax.set_xscale('log')                    
            
ax.set(xlabel = '$\sqrt{n_{dof}}$',ylabel = 'L2 err')
ax.legend(loc = 'best')

#plt.gca().invert_xaxis()
plt.ylim((1e-4,1))
plt.xlim((10,200))
plt.show()


fig.savefig('L2_err_struct')