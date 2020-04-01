#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Sun Feb 16 10:37:11 2020

@author: simo94


This script computes the L2 deviations from a steady state solution of the nonlinear 2D Shallow Water equations.
The equations are solved using a Mixed Finite Element method. 

Convergence to the numerical steady state solution is obtained by time integration using a RK4 scheme.


The initial state takes the form: 
    
    h = 10 + 1/(4*pi)*cos(4*pi*x[1]))
    u = (sin(4*pi*x[1]),0.0)
    
    % Absolute vorticity
    q = (-4*pi*cos(4*pi*x[1]) + 10.0)/10 + 1/(4*pi)*cos(4*pi*x[1]))


Domain: Unit square domain with periodic boundary conditions.
   
The solution is computed in solver_SW.py 

The module refinement.py returns a structured or unstructured mesh refined
locally using a gradient-based monitor function.


At each time step interpolate solution to the new mesh or solve directly onto
the equidistributed mesh ?

1st attempt: Try to solve equation directly onto the equidistributed mesh 
from the mov_mesh module 

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

                            
def conv_rate(xvalues,err):

    'Compute convergence rate '    
    
    l = xvalues.shape[0]
    rate_h = np.zeros(l-1)
    rate_u = np.zeros(l-1)
    rate_q = np.zeros(l-1)
    
    
    for i in range(l-1):
        rate_u[i] = ln(err[i,0]/err[i+1,0])/ln(sqrt(xvalues[i+1,0]/xvalues[i,0]))
        rate_h[i] = ln(err[i,1]/err[i+1,1])/ln(sqrt(xvalues[i+1,1]/xvalues[i,1]))
        rate_q[i] = ln(err[i,2]/err[i+1,2])/ln(sqrt(xvalues[i+1,2]/xvalues[i,2]))
        
    rate_u = rate_u[-1]
    rate_h = rate_h[-1]
    rate_q = rate_q[-1]
    
    print('convergence rate of u is ' + str(rate_u))
    print('convergence rate of h is ' + str(rate_h))
    print('convergence rate of q is ' + str(rate_q))
    

    return rate_u,rate_h,rate_q



#his step might rise issues if we want to ensure conservation of Mass, Absolute Vorticity, Energy, Enstrophy.
N  = np.array([10,20,25,30,40,50,60])
n_iter = N.shape[0]

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
    
space_str = 'CG2BDMF2DG1'

for i in range(n_iter):
    
    print('step ',i+1)
    
    mesh = UnitSquareMesh(N[i],N[i]) 
    #mesh = RectangleMesh.create([Point(0.0,0.0),Point(1.0,1.0)],[N[i],N[i]],CellType.Type.quadrilateral)
    mesh = equid_mesh(N[i],mesh,source_dx_str,alpha,n_equid_iter,arc_length=1)
    
    ## plot refined mesh is interested
    #plt.figure(i)
    #plot(mesh)
    #plt.title('Mesh_N_' + str(N[i]))

    # Compute CFL condition 
    cfl = dt/mesh.hmin()
    
    if cfl > 1.0:
        print('cfl condition is not satisfied')
    

    # Continuous Lagrange Vector function space
    E = FiniteElement('CG',mesh.ufl_cell(),1)


    U = FiniteElement('RT',mesh.ufl_cell(),1)

    # Free surface perturbation field' function spacea
    H = FiniteElement('DG',mesh.ufl_cell(),0)

    
    W_elem = MixedElement([U,H])
    W1 = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())
    
    W_elem = MixedElement([E,U])
    W2 = FunctionSpace(mesh,W_elem,constrained_domain=PeriodicBoundary())

    
    # Return solution u,h at the final time step
    # error_vec contains the deviations from initial condition over time 

    u,h,vort,error_vec,scalars = solver(mesh,W1,W2,dt,tf,output=0,lump=0)

    # dof = W1.dim()
    dof_h = h.vector().size()
    dof_u = u.vector().size()
    dof_vort = vort.vector().size()
   
    
    err[i,:] = error_vec[-1,:]        
    
    # error with u_e and h_e, n 
    dof[i] = [dof_u,dof_h,dof_vort]
    

#### Deviations from initial condition ####
    
    
# Save deviations 
str_file = 'data/dev_N_' + str(N[-1])  + space_str
np.save(str_file,error_vec)        
   
# Plot oscillatory deviations from initial condition over time 
fig, ax = plt.subplots()
fig.tight_layout()
ax.plot(error_vec[:,0],label = 'u')
ax.plot(error_vec[:,1],label = 'h')
#ax.plot(error_vec[:,2],label = 'q')
ax.set_xlabel('t')
ax.legend(loc = 'best')
plt.title('deviations_N_' + str(N[-1])  + space_str)



#### Physical invariants ####

# Rescale scalar variables and plot deviations for last N 

scalars_norm = scalars 

for i in range(scalars.shape[1]):
    scalars_norm[:,i] = (scalars[:,i] - scalars[0,i])/scalars[0,i]

# Save physical quantities 
str_file = 'data/scalars_N_' + str(N[-1])  + space_str


np.save(str_file,scalars)        

# Plot Scalar quantities and check for convergence
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



#### Convergence Rate ####


# Compute convergence rate
rate_u,rate_h,rate_q = conv_rate(dof,err)

str_file = 'data/conv_N_' + str(N[-1])  + space_str
np.save(str_file,error_vec)        
fig, ax = plt.subplots()
ax.plot(np.sqrt(dof[:,0]),err[:,0],linestyle = '-.',marker = 'o',label = 'u:'+ "%.4g" %rate_u)
ax.plot(np.sqrt(dof[:,1]),err[:,1],linestyle = '-.',marker = 'o',label = 'h:'+ "%.4g" %rate_h)
#ax.plot(np.sqrt(dof[:,2]),err[:,2],linestyle = '-.',marker = 'o',label = 'q:'+ "%.4g" %rate_q)
ax.set_xlabel('$\sqrt{n_{dof}}$')
ax.set_ylabel('deviations')

ax.set_yscale('log')
ax.set_xscale('log')           
ax.legend(loc = 'best')
plt.title('Convergence_rate_N_'+ str(N[-1])  + space_str)


