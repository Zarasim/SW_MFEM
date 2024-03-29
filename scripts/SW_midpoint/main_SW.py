#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Sun Feb 16 10:37:11 2020

@author: Zarasim

This script computes the L2 deviations from a steady state solution of the nonlinear 2D Shallow Water equations.
The equations are solved using the Mixed Finite Element method on a periodic unit square domain. 

Convergence to the numerical steady state solution is obtained through time integration with RK4 scheme.


The initial condition for h and u in the 1D case is: 
    
    h = 10 + 1/(4*pi)*cos(4*pi*x[1]))
    u = (sin(4*pi*x[1]),0.0)
   
    
The solution is computed in solver_SW.py 

The module refinement.py returns a structured or unstructured mesh refined
locally using a gradient-based monitor function.


"""



# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from SW_midpoint import *

from mov_mesh import *
from mshr import *
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
                            
def conv_rate(xvalues,err):

    'Compute convergence rate '    
    
    l = xvalues.shape[0]
    rate_u = np.zeros(l-1)
    rate_h = np.zeros(l-1)
    
    
    for i in range(l-1):
        rate_u[i] = ln(err[i,0]/err[i+1,0])/ln(sqrt(xvalues[i+1,0]/xvalues[i,0]))
        rate_h[i] = ln(err[i,1]/err[i+1,1])/ln(sqrt(xvalues[i+1,1]/xvalues[i,1]))
        
    rate_u = rate_u[-1]
    rate_h = rate_h[-1]

    print('convergence rate of u is ' + str(rate_u))
    print('convergence rate of h is ' + str(rate_h))
    

    return rate_u,rate_h


# For Delaunay tesselation do not use too small numbers or the solver diverges
N  = np.array([20])

# N for right-based and crossed triangles
# 10,20,30,40,60
# N for unstructured mesh
#12,15,21,25,28
n_iter = N.shape[0]


# Store err and dof
err_sol = np.zeros(2*n_iter).reshape(n_iter,2)
dof = np.zeros(2*n_iter).reshape(n_iter,2)
dev_scalars = np.zeros(4*n_iter).reshape(n_iter,4)
dof_tot = np.zeros(n_iter)

### Parameters for equidistributing mesh ###    

n_equid_iter = 5
alpha = 10

# Choose monitor function 
source_dx_str = 'np.sqrt((4*pi*np.cos(4*pi*x))**2 + (np.sin(4*pi*x))**2)'
#'4*pi*np.cos(4*pi*x)'

dt = 0.01
tf = 50.0
nt = np.int(tf/dt)

t_vec = np.arange(1,nt+2)*dt

space_str = 'CG1RT1DG0_adapt'


for i in range(n_iter):
    
    print('step ',i+1)
    
    mesh = UnitSquareMesh(N[i],N[i]) 
    ## Quadrilateral mesh does not support RT,BDM space and refinement
    #mesh = RectangleMesh.create([Point(0.0,0.0),Point(1.0,1.0)],[N[i],N[i]],CellType.Type.quadrilateral)
    
    ## Generate unstructured mesh with Delaunay triangulation 
    #domain = Rectangle(Point(0.,0.),Point(1.,1.))
    #mesh = generate_mesh(domain,N[i],'cgal') 
    #mesh.smooth()
    
    ## Equidistribute mesh
    #mesh = equid_mesh(N[i],mesh,source_dx_str,alpha,n_equid_iter,arc_length=1)
       
    # Plot refined mesh 
    plt.figure(i)
    plot(mesh)
    
    E = FiniteElement('CG',mesh.ufl_cell(),1)

    U = FiniteElement('RT',mesh.ufl_cell(),1)

    H = FiniteElement('DG',mesh.ufl_cell(),0)

       
    # Define Mixed Function space
    W_elem = MixedElement([E,U,U,H])
    W = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())    
    
    # devs contains deviations from initial condition over time
    # scalars contains deviations of physical quantities over time 
    dev_sol,dev_scalars = solver(mesh,W,dt,tf,output = None,lump = None)
    
    dof_u = W.sub(1).dim()
    dof_h = W.sub(3).dim()
    dof[i] = [dof_u,dof_h]
    
    for j in range(2):     
        err_sol[i,j] = np.mean(dev_sol[:,j])             
        

rel_path = os.getcwd()
pathset = os.path.join(rel_path,'Data_' + space_str)
if not(os.path.exists(pathset)):
    os.mkdir(pathset)


#### Deviations solution ####
        
# Save deviations 
output_file = 'dev_sol_N_' + str(N[-1])  + '_' + space_str
np.save(os.path.join(pathset, output_file),dev_sol)        

   
# Plot oscillatory deviations from initial condition over time 
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
plt.title('h')
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))

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
plt.title('Energy')
plt.ylabel('$(E - E_{0})/E_{0}$')
plt.xlabel('t')
plt.subplot(2,2,2)
plt.plot(t_vec,scalars_norm[:,1])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Enstrophy')
plt.xlabel('t')
plt.ylabel('$(Q - Q_{0})/Q_{0}$')
plt.subplot(2,2,3)
plt.plot(t_vec,scalars_norm[:,2])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Absolute vorticity')
plt.ylabel('$(q - q_{0})/q_{0}$')
plt.xlabel('t')
plt.subplot(2,2,4)
plt.plot(t_vec,scalars_norm[:,3])
plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
plt.title('Mass')
plt.ylabel('$(m - m_{0})/m_{0}$')
plt.xlabel('t')
fig.tight_layout()

#### Convergence Rate ####


# Compute convergence rate
# rate_E,rate_Ens = conv_rate(dt,err_scalars)

# output_file = 'err_scalars' + str(N[-1])  +  '_' + space_str
# np.save(os.path.join(pathset, output_file),err_sol)        


# output_file = 'dt_vec' + str(N[-1])  + '_' + space_str
# np.save(os.path.join(pathset, output_file),dof)        


# fig, ax = plt.subplots()
# ax.plot(np.sqrt(dof[:,0]),err_sol[:,0],linestyle = '-.',marker = 'o',label = 'Energy:'+ "%.4g" %rate_E)
# ax.plot(np.sqrt(dof[:,1]),err_sol[:,1],linestyle = '-.',marker = 'o',label = 'Enstrophy:'+ "%.4g" %rate_Ens)
# ax.set_xlabel('$\sqrt{n_{dof}}$')
# ax.set_ylabel('L2 error')
# ax.set_yscale('log')
# ax.set_xscale('log')           
# ax.legend(loc = 'best')

