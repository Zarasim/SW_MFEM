#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Sun Feb 16 10:37:11 2020

@author: simo94


This script simulates a merging vortex dynamics by solving the 2D nonlinear
rotaring Shallow water equations.

input: Gaussin radially symmetric streamfunction psi


u = (dy_psi,dx_psi)
h = psi

f = g = 10.0

Domain: Unit square domain with periodic boundary conditions.


The given initial conditions are in quasi-geostrophic balance if 
inertial forces are neglected. 

   
The solution is computed in solver_SW.py 

The module refinement.py returns a structured or unstructured mesh refined
locally using a gradient-based monitor function.



"""

# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from SW_RK4 import *
from mov_mesh import *
import matplotlib.pyplot as plt

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



N  = 30

# Store err and dof
err = np.zeros(3*n_iter).reshape(n_iter,3)
dof = np.zeros(3*n_iter).reshape(n_iter,3)


### Parameters for equidistributing mesh ###    
n_equid_iter = 5
alpha = 10

# string expression for derivative of velocity field
source_dx_str = 'np.sqrt((4*pi*np.cos(4*pi*x))**2 + (np.sin(4*pi*x))**2)'
#'4*pi*np.cos(4*pi*x)'
#'np.sqrt((4*pi*np.cos(4*pi*x))**2 + (np.sin(4*pi*x))**2)'

dt = 0.0005
tf = 0.1
    
space_str = 'CG1RT1DG0'

mesh = UnitSquareMesh(N,N) 
#mesh = equid_mesh(N[i],mesh,source_dx_str,alpha,n_equid_iter,arc_length=1)
    
## plot refined mesh is interested
#plt.figure(i)
#plot(mesh)
#plt.title('Mesh_N_' + str(N[i]))
    

# Continuous Lagrange Vector function space
E = FiniteElement('CG',mesh.ufl_cell(),1)
    
U = FiniteElement('RT',mesh.ufl_cell(),1)
    
# Free surface perturbation field' function spacea
H = FiniteElement('DG',mesh.ufl_cell(),0)
    
        
W_elem = MixedElement([U,H])
W1 = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())
        
W_elem = MixedElement([E,U])
W2 = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())

    
# error_vec contains the deviations from initial condition over time 

error_vec,scalars = solver(mesh,W1,W2,dt,tf,output=1,lump=0)


#### Deviations from initial condition ####
    
    
# Save deviations 
str_file = 'data/dev_N_' + str(N[-1])  + space_str
np.save(str_file,error_vec)        
 
  
# Plot oscillatory deviations from initial condition over time 
fig, ax = plt.subplots()
fig.tight_layout()
ax.plot(error_vec[:,0],label = 'q')
ax.plot(error_vec[:,1],label = 'u')
ax.plot(error_vec[:,2],label = 'h')

ax.set_xlabel('t')
ax.legend(loc = 'best')



#### Physical invariants ####

# Rescale scalar variables and plot deviations for last N 

scalars_norm = scalars 

for i in range(scalars.shape[1]):
    scalars_norm[:,i] = (scalars[:,i] - scalars[0,i])/scalars[0,i]

# Save physical quantities 
str_file = 'data/scalars_N_' + str(N[-1])  + space_str


np.save(str_file,scalars)        
# Plot Scalar quantities
fig = plt.figure(n_iter + 3)
plt.subplot(2,2,1)
plt.plot(scalars_norm[:,0])
plt.ticklabel_format(axis="y", style="sci")
plt.title('Energy')
plt.subplot(2,2,2)
plt.plot(scalars_norm[:,1])
plt.ticklabel_format(axis="y", style="sci")
plt.title('Enstrophy')
plt.subplot(2,2,3)
plt.plot(scalars_norm[:,2])
plt.ticklabel_format(axis="y", style="sci")
plt.title('Absolute vorticity')
plt.subplot(2,2,4)
plt.plot(scalars_norm[:,3])
plt.ticklabel_format(axis="y", style="sci")
plt.title('Mass')
fig.tight_layout()
