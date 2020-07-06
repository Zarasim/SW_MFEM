#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:30:42 2020


This module computes the equidistributed mesh for the Steady State solution
of the Shallow Water problem. The a-priori knowledge of the exact solution is 
used for computing the monitor function rho.

The equidistribution condition is given by De Boor's algorithm.

Optimality of the mesh is given in terms of piece-wise constant or 
linear interpolation. This argument ignores possible errors of the solution 
at the mesh nodes. See  Huang-Russell [1] for the derivation. 

For MFEM, FEM method, the residual upper bound is given by the interpolation 
error. 

This upper bounds depend on two conditions: order of interpolant (k) and norm 
evaluation (m)


@author: simo94
"""

# use MATLAB code provided by Thomas Melvin for Shallow Water


from dolfin import *
import numpy as np

import matplotlib.pyplot as plt

def monitor(x,sol_1,sol_2,alpha,beta,arc_length):
    
    
    # Use a priori information for computing the derivative at each point         
    m = np.sqrt(1 + (alpha*sol_1)**2 + (beta**sol_2)**2 )
        
    # normalize
    m = m/np.sum(m)
    
    return m


def equidistribute(x,rho):
    
    
    # Make a copy of vector x to avoid overwriting
    y = x.copy()
    
    # number of mesh points counted from 0 ton nx-1
    nx = x.shape[0]
    

    II = nx - 1 
    JJ = nx - 1
    
    # Create vector of integrals with nx entries 
    intMi = np.zeros(nx)
    
    
    # compute each integral using trapezoidal rule
    intMi[1:] = 0.5*(rho[1:] + rho[:-1])*np.diff(x)
    
    # take cumulative sum of integrals
    intM = np.cumsum(intMi)
    
    # take total integral theta
    theta = intM[-1]
    
    
    jj = 0
    
    # Assign new nodes from  y_1 to y_(nx - 2)
    for ii in range(1,II):
        
        # Target =  y_1 = 1/(nx-1)*theta ... y_nx-2 = (nx-2)/(nx-1)*theta
    
        Target = ii/II*theta
        
    
        while jj < JJ and intM[jj] < Target:
        
            jj = jj+1
            
        jj = jj - 1
        
        Xl = x[jj]
        Xr = x[jj+1]
        Ml = rho[jj]
        Mr = rho[jj+1]
        
        Target_loc = Target - intM[jj]
        
        mx = (Mr - Ml)/(Xr - Xl)
        
        y[ii] = Xl + 2*Target_loc/(Ml + np.sqrt(Ml**2 + 2*mx*Target_loc))
        
        
    return y



def equid_mesh(N,mesh,source,alpha,beta,n_iter,arc_length=1):
    
    
    # equidistribute along y direction 
    y = mesh.coordinates()[:,1][0::N+1]
    
  
    source_u = eval('lambda x: ' + source[0])
    source_h = eval('lambda x: ' + source[1])

    for i in range(n_iter):
        
        sol1 = source_u(y)
        sol2 = source_h(y)
        rho = monitor(y,sol1,sol2,alpha,beta,1)
    
        # Use De Boor algorithm 
        y = equidistribute(y,rho)
    
    
    y = np.repeat(y,N+1)
    mesh.coordinates()[:,1] = y

    return mesh 