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
from SW_RK4 import *
#from adaptive_mesh_1D import *
from Winslow import *
from mshr import *
from initial_fields import *
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

dt = 0.0005
tf = 0.01
nt = np.int(tf/dt)

t_vec = np.arange(1,nt+2)*dt


space_q = 'CG'
deg_q = 1
space_u = 'RT'
deg_u = 1
space_h = 'DG'
deg_h = 0


test_dim = '1D'

space_str = space_q + str(deg_q) + space_u + str(deg_u) + space_h + str(deg_h) 


for i in range(n_iter):
    
    print('step ',i+1)
    
    mesh = UnitSquareMesh(N[i],N[i]) 
    
    ## Quadrilateral mesh does not support RT,BDM space and refinement
    #mesh = RectangleMesh.create([Point(0.0,0.0),Point(1.0,1.0)],[N[i],N[i]],CellType.Type.quadrilateral)
    
    ## Generate unstructured mesh with Delaunay triangulation 
    #domain = Rectangle(Point(0.,0.),Point(1.,1.))
    #mesh = generate_mesh(domain,N[i],'cgal') 
    #mesh.smooth()
    
    
    ## Set up initial exact condition 
    u_0,h_0 = initial_fields(mesh,space_u,space_h,deg_u,deg_h,test_dim)

    
    ## Plot initial equidistributed mesh
    #mesh = equid_mesh(N[i],mesh,source_dx_str,alpha,n_equid_iter,arc_length=1)
    #mesh = Winslow_eq(mesh,N[i],u_0,h_0,monitor = 'arc-length')   
    
    
    # Plot refined mesh 
    #plt.figure(i)
    #plot(mesh)
             
    E = FiniteElement(space_q,mesh.ufl_cell(),deg_q)
    U = FiniteElement(space_u,mesh.ufl_cell(),deg_u)
    H = FiniteElement(space_h,mesh.ufl_cell(),deg_h)

    
    W_elem = MixedElement([U,H])
    W1 = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())
    
    W_elem = MixedElement([E,U])
    W2 = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())
    
    
    # dev_sol contains devs from u_0 and h_0 in t
    # dev_scalars contains devs of Energy,Enstrophy,q,Mass in t
    dev_sol,dev_scalars = solver(mesh,W1,W2,u_0,h_0,dt,tf,output=0,lump=0,case = test_dim) 

    dof_tot[i] = W1.dim()
    dof_u = W1.sub(0).dim()
    dof_h = W1.sub(1).dim()
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
rate_u,rate_h = conv_rate(dof,err_sol)

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

