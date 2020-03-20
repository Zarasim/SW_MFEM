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


Domain: Unit square domain with periodic boundary conditions.
   
The solution is computed in solver_SW.py 

The module refinement.py returns a structured or unstructured mesh refined
locally using a gradient-based monitor function.

"""


# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from SW_RK4 import *
from refinement import *


import matplotlib.pyplot as plt


# Number of mesh points is N+1 for each side of the square
#N  = np.array([8,10,15,20,25,40,60])

# In case of refinement, start with a fixed number of nodes and
# increase the refinement levels
n_iter = 3


# Store err and dof
err = np.zeros(2*n_iter).reshape(n_iter,2)
dof = np.zeros(2*n_iter).reshape(n_iter,2)



# Set periodic boundary conditions 
class PeriodicBoundary(SubDomain):

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
    
    
    l = xvalues.shape[0]
    rate_h = np.zeros(l-1)
    rate_u = np.zeros(l-1)
    
    
    for i in range(l-1):
        rate_u[i] = ln(err[i,0]/err[i+1,0])/ln(xvalues[i+1,0]/xvalues[i,0])
        rate_h[i] = ln(err[i,1]/err[i+1,1])/ln(xvalues[i+1,1]/xvalues[i,1])
    
    print('mean convergence rate of u is ' + str(np.mean(rate_u)))
    print('mean convergence rate of h is ' + str(np.mean(rate_h)))


# Generate unstructured triangular mesh 
#domain = Rectangle(Point(0.,0.),Point(1.0,1.0))
#mesh = generate_mesh(domain,N) 


N  = 15
mesh = UnitSquareMesh(N,N)


for i in range(n_iter):
    
    print('step ',i+1)
    
    
    mesh = refinement(mesh,n_ref = i,ref_ratio = 0.6)
    
    ## plot refined mesh is interested
    fig = plt.figure(figsize=(10,10))
    plot(mesh)
    
    dt = 0.0001
    
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
    # deviations explode for refinement >=2  
    
    u,h,error_vec = solver(mesh,W1,W2,dt,lump=0)

    
    # Plot oscillatory deviations from initial condition over time 
    plt.subplot(2,1,1)
    plt.plot(error_vec[:,0])
    plt.title('dev u')
    plt.subplot(2,1,2)
    plt.plot(error_vec[:,1])
    plt.title('dev h')
    
    dof_h = h.vector().size()
    dof_u = u.vector().size()
    
    err[i,:] = error_vec[-2,:]        
    
    # error with u_e and h_e, n 
    dof[i] = [dof_u,dof_h]
    
    
    
err_array = np.array([err,dof])

# Compute convergence rate
conv_rate(dof,err)


# Save error and dof in a npy file 
#np.save('SWRK4_ref_err',err_array)        

    