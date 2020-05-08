#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:41:27 2020

@author: simo94
"""



## To DO next:

# Compute cell value of function in DG space
# Add smoothing filter for w


# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from dolfin import *
from dolfin.cpp.mesh import MeshQuality

import matplotlib.pyplot as plt
import numpy as np


def Winslow_eq(mesh,N,u,h,monitor = 'arc-length'):
    
    num_mesh_vertices = (N + 1) ** 2
    mesh_vertex_coords = mesh.coordinates()

    
    # space of vertex coordinates    
    V = FunctionSpace(mesh,'CG',1)
    W = FunctionSpace(mesh,'CG',1)
 

    #  assign initial condition for u in the physical space, which coincides with the computational one
    # u.interpolate(Expression('10.000 + sin(2*pi*x[0])*sin(pi*x[1])',element = U.ufl_element()))
    # u2.interpolate(Expression('sin(4.0*pi*x[1])',element = U.ufl_element())) 
    
    #w = project(0.5*(sqrt((u[0].dx(0)*u[0].dx(0) + u[1].dx(1)*u[1].dx(1))) + sqrt(h.dx(0)*h.dx(0) + h.dx(1)*h.dx(1))),W)
    beta = 1.0
    w = project(sqrt(1 + beta*(h.dx(0)*h.dx(0) + h.dx(1)*h.dx(1)) + beta*(u[0].dx(0)*u[0].dx(0) + u[0].dx(1)*u[0].dx(1))),W)
    
    x_D = Expression('x[0]',degree=2)
    y_D = Expression('x[1]',degree=2)
    
    bc_x = DirichletBC(V,x_D,'on_boundary')
    bc_y = DirichletBC(V,y_D,'on_boundary')
    
    x_trial = TrialFunction(V)
    y_trial = TrialFunction(V)
    
    x_test = TestFunction(V)
    y_test = TestFunction(V)
    
    x = Function(V)
    y = Function(V)
    
    n = FacetNormal(mesh)
    
    a = x_test*w*inner(grad(x_trial),n)*ds - inner(grad(x_test),w*grad(x_trial))*dx
    L = Constant(0.0)*x_test*dx
    
    solve(a==L,x,bc_x)
    
    a = y_test*w*inner(grad(y_trial),n)*ds - inner(grad(y_test),w*grad(y_trial))*dx
    L = Constant(0.0)*y_test*dx
    
    solve(a==L,y,bc_y)
    
    
    mesh.coordinates()[:,0] = x.compute_vertex_values()
    mesh.coordinates()[:,1] = y.compute_vertex_values()
    
    return mesh
   





