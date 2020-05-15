#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb 16 10:37:11 2020

@author: Zarasim

Solve the linear Rotating Shallow Water equations

The initial perturbation field about a rest state is 

    h = 10 + 1/(4*pi)*cos(4*pi*x[1]))
    u = sin(4*pi*x[1])
    v = 0
    
This condition satisfies the geostrophic balance. The equation is then integrated in time 
using a RK4 scheme. Deviations from the initial condition are given by the propagation of 
inertia-gravity waves. For a particular combination of mesh resolution and time step, we can observe
additional spurious oscillations with frequency w = O(N). These can be ignored whilst still preserving
Mass and Energy. 

The solution is computed in solver_SW.py 

## Refinement Stategy ## 

The refinement module returns a structured adapted mesh using a solution gradient-based monitor function.

"""

# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from SW_RK4 import *
from adaptive_mesh_1D import *
from mshr import *
from initial_fields import *
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import os 


class PeriodicBoundary(SubDomain):
    
    ' Set periodic boundary conditions '

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.
            
            
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
    print('Maximum angular frequency are: ',(2*np.pi)*freq[1 + idx[0]])            
 

           
def conv_rate(dof,err):

    ' Compute convergence rate '   
    l = dof.shape[0]
    rate = np.zeros(l-1)
    
    for i in range(l-1):
        rate[i] = ln(err[i]/err[i+1])/ln(sqrt(dof[i+1]/dof[i]))
        
    rate = np.mean(rate)

    print('Convergence rate is ' + str(rate))

    return rate

N  = np.array([10,20,30,40,50])


# N for unstructured mesh
#12,15,21,25,28
n_iter = N.shape[0]


# Store err and dof
err_sol = np.zeros(2*n_iter).reshape(n_iter,2)
dof = np.zeros(2*n_iter).reshape(n_iter,2)
dof_tot = np.zeros(n_iter)


dt = 0.0005
tf = 0.5
nt = np.int(tf/dt)
t_vec = np.arange(1,nt+2)*dt


space_u = 'RT'
deg_u = 1
space_h = 'DG'
deg_h = 0


test_dim = '1D'
space_str = space_u + str(deg_u) + space_h + str(deg_h) 


## Parameters for equidistributing mesh
n_equid_iter = 5
alpha = 1.0
beta = 1.0
source_str = ['2*pi*np.cos(2.00*pi*x)','np.sin(2.0000*pi*x)']


for i in range(n_iter):
    
    print('step ',i+1)
    
    mesh = UnitSquareMesh(N[i],N[i]) 
    
    ## Quadrilateral mesh does not support RT,BDM space and refinement
    #mesh = RectangleMesh.create([Point(0.0,0.0),Point(1.0,1.0)],[N[i],N[i]],CellType.Type.quadrilateral)
    
    ## Generate unstructured mesh with Delaunay triangulation 
    #domain = Rectangle(Point(0.,0.),Point(1.,1.))
    #mesh = generate_mesh(domain,N[i],'cgal') 
    #mesh.smooth()
    
    
    ## Equidistributing mesh with respect to gradient of solution 
    #mesh = equid_mesh(N[i],mesh,source_str,alpha,beta,n_equid_iter,arc_length=1)
    #fig = plt.figure()
    #plot(mesh)
    
    ## Set up initial exact condition 
    u_0,h_0 = initial_fields(mesh,space_u,space_h,deg_u,deg_h,test_dim)
   
   
    ## Create FE spaces          
    U = FiniteElement(space_u,mesh.ufl_cell(),deg_u)
    H = FiniteElement(space_h,mesh.ufl_cell(),deg_h)

    
    W_elem = MixedElement([U,H])
    W = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())
    
    
    # dev_sol contains solution u_0 and h_0 
    # dev_scalars contains devs of Energy,Enstrophy,q,Mass in t
    sol,dev_sol,dev_scalars = solver(mesh,W,u_0,h_0,dt,tf,output=0,lump=0,case = test_dim)

    
    dof_tot[i] = W.dim()
    dof_u = W.sub(0).dim()
    dof_h = W.sub(1).dim()
    dof[i] = [dof_u,dof_h]
    
    for j in range(2):     
        err_sol[i,j] = np.mean(dev_sol[:,j])       
        
        ## Store err_scalars for simulations with different delta t
        #err_scalars[i,j] = np.mean(dev_scalars[:,j])
    
rel_path = os.getcwd()
pathset = os.path.join(rel_path,'Data_' + space_str)
if not(os.path.exists(pathset)):
    os.mkdir(pathset)


#### Plot solution in fixed point t ####
        
# Save deviations 
output_file = 'sol_N_' + str(N[-1])  + '_' + space_str
np.save(os.path.join(pathset, output_file),sol)        

   
# Plot oscillatory deviations from initial condition over time 
fig = plt.figure()
plt.plot(t_vec,sol[:,0])
plt.xlabel('t')
plt.ylabel('u')
#plt.title('u')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.savefig('u')
fig = plt.figure()
plt.plot(t_vec,sol[:,1])
plt.xlabel('t')
plt.ylabel('v')
#plt.title('v')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.savefig('v')
fig = plt.figure()
plt.plot(t_vec,sol[:,2])
plt.xlabel('t')
plt.ylabel('h')
#plt.title('h')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.savefig('h')


# Compute Fourier transform of the quantities 
Fourier_trans(dt,sol[:,0])
Fourier_trans(dt,sol[:,1])
Fourier_trans(dt,sol[:,2])



####  Deviations Physical quantities ####


# Save scalar deviations
output_file = 'dev_scalars_N_' + str(N[-1]) + '_' + space_str
np.save(os.path.join(pathset, output_file),dev_scalars)     

# Rescale scalar variables and plot deviations for highest dof

scalars_norm = dev_scalars 
for i in range(dev_scalars.shape[1]):
    scalars_norm[:,i] = (dev_scalars[:,i] - dev_scalars[0,i])/dev_scalars[0,i]

   
 # Plot Scalar quantities and check for convergence
fig = plt.figure()
plt.subplot(2,2,1)
plt.plot(t_vec,scalars_norm[:,0])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Potential Energy')
plt.ylabel('$(E - E_{0})/E_{0}$')
plt.xlabel('t')
plt.subplot(2,2,2)
plt.plot(t_vec,scalars_norm[:,1])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Kinetic Energy')
plt.xlabel('t')
plt.ylabel('$(E - E_{0})/E_{0}$')
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




#### Plot L2-dev ###
        
# Save deviations 
output_file = 'dev_sol_N_' + str(N[-1])  + '_' + space_str
np.save(os.path.join(pathset, output_file),dev_sol)        

   
# Plot oscillatory deviations from initial condition over time 
fig = plt.figure()
plt.plot(t_vec,dev_sol[:,0])
plt.xlabel('t')
plt.ylabel('velocity')
#plt.title('u')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.savefig('L2_dev_u')
fig = plt.figure()
plt.plot(t_vec,dev_sol[:,1])
plt.xlabel('t')
plt.ylabel('h')
#plt.title('h')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.savefig('L2_dev_h')



#### Convergence Rate ####


# Compute convergence rate for solution 
rate_u = conv_rate(dof[:,0],err_sol[:,0])
rate_h = conv_rate(dof[:,1],err_sol[:,1])

output_file = 'err' + str(N[-1])  +  '_' + space_str
np.save(os.path.join(pathset, output_file),err_sol)        

output_file = 'dof' + str(N[-1])  + '_' + space_str
np.save(os.path.join(pathset, output_file),dof)        

fig, ax = plt.subplots()
ax.plot(np.sqrt(dof[:,0]),err_sol[:,0],linestyle = '-.',marker = 'o',label = 'u:'+ "%.4g" %rate_u)
ax.plot(np.sqrt(dof[:,1]),err_sol[:,1],linestyle = '-.',marker = 'o',label = 'h:'+ "%.4g" %rate_h)
ax.set_xlabel('$\sqrt{n_{dof}}$')
ax.set_ylabel('L2 error')
ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')
plt.savefig('conv_rate')

