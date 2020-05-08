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


                    
def conv_rate(xvalues,err):

    'Compute convergence rate '    
    
    l = xvalues.shape[0]
    rate_u = np.zeros(l-1)
    rate_h = np.zeros(l-1)
    
    
    for i in range(l-1):
        rate_u[i] = np.log(err[i,0]/err[i+1,0])/np.log(np.sqrt(xvalues[i+1,0]/xvalues[i,0]))
        rate_h[i] = np.log(err[i,1]/err[i+1,1])/np.log(np.sqrt(xvalues[i+1,1]/xvalues[i,1]))
        
    rate_u = rate_u[-1]
    rate_h = rate_h[-1]

    print('convergence rate of u is ' + str(rate_u))
    print('convergence rate of h is ' + str(rate_h))
    

    return rate_u,rate_h



# get home directory from python 

os.getenv("HOME")

data_path = '/home/simo94/Documents/Data_folder'
fin_path = '/RK4/Uniform/2D/Data_CG2BDM1DG0'



pathset = os.path.join(data_path + fin_path)
outputfile1 = '/home/simo94/Documents/Data_folder/RK4/Uniform/2D/Data_CG2BDM1DG0/err60_CG2BDM1DG0.npy'
outputfile2 = '/home/simo94/Documents/Data_folder/adaptive/2D/Data_CG2BDM1DG0/err60_CG2BDM1DG0.npy'


#outputfile2 = '/dev_sol_N_10_CG1RT1DG0.npy'
#outputfile3 = '/err20_CG1RT1DG0.npy'
#outputfile4 = '/dof20_CG1RT1DG0.npy'



err_sol = np.load(outputfile1)
err_sol_adap = np.load(outputfile2)
dof = np.load('/home/simo94/Documents/Data_folder/adaptive/2D/Data_CG2BDM1DG0/dof60_CG2BDM1DG0.npy')

#dev_scalars = np.load(pathset+outputfile1)
#dev_sol = np.load(pathset+outputfile2)
#err_sol = np.load(pathset+outputfile3)



# dt = 0.0005
# tf = 0.01
# nt = np.int(tf/dt)

# t_vec = np.arange(1,nt+2)*dt

# #Rescale scalar variables and plot deviations for highest dof

# scalars_norm = dev_scalars 
# for i in range(dev_scalars.shape[1]):
#     scalars_norm[:,i] = (dev_scalars[:,i] - dev_scalars[0,i])/dev_scalars[0,i]

   
# # Plot Scalar quantities and check for convergence
# fig = plt.figure()
# plt.subplot(2,2,1)
# plt.plot(t_vec,scalars_norm[:,0])
# plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# plt.title('Energy')
# plt.ylabel('$(E - E_{0})/E_{0}$')
# plt.xlabel('t')
# plt.subplot(2,2,2)
# plt.plot(t_vec,scalars_norm[:,1])
# plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# plt.title('Enstrophy')
# plt.xlabel('t')
# plt.ylabel('$(Q - Q_{0})/Q_{0}$')
# plt.subplot(2,2,3)
# plt.plot(t_vec,scalars_norm[:,2])
# plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# plt.title('Absolute vorticity')
# plt.ylabel('$(q - q_{0})/q_{0}$')
# plt.xlabel('t')
# plt.subplot(2,2,4)
# plt.plot(t_vec,scalars_norm[:,3])
# plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# plt.title('Mass')
# plt.ylabel('$(m - m_{0})/m_{0}$')
# plt.xlabel('t')
# fig.tight_layout()



# #### Deviations solution ####
        
   
# # Plot oscillatory deviations from initial condition over time 
# fig = plt.figure()
# plt.plot(t_vec,dev_sol[:,0])
# plt.xlabel('t')
# plt.ylabel('L2 error')
# plt.title('u')
# plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# fig = plt.figure()
# plt.plot(t_vec,dev_sol[:,1])
# plt.xlabel('t')
# plt.ylabel('L2 error')
# plt.title('h')
# plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))



# Compute convergence rate
rate_u,rate_h = conv_rate(dof,err_sol)
rate_u_adap,rate_h_adap = conv_rate(dof,err_sol_adap)
   

fig, ax = plt.subplots()
#ax.plot(np.sqrt(dof[:,0]),err_sol[:,0],linestyle = '-.',marker = 'o',label = 'u:'+ "%.4g" %rate_u)
ax.plot(np.sqrt(dof[:,1]),err_sol[:,1],linestyle = '-.',marker = 'o',label = 'h:'+ "%.4g" %rate_h)
#ax.plot(np.sqrt(dof[:,0]),err_sol_adap[:,0],linestyle = '-.',marker = 'o',label = 'u_adap:'+ "%.4g" %rate_u_adap)
ax.plot(np.sqrt(dof[:,1]),err_sol_adap[:,1],linestyle = '-.',marker = 'o',label = 'h_adaptive:'+ "%.4g" %rate_h_adap)
ax.set_xlabel('$\sqrt{n_{dof}}$')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')


