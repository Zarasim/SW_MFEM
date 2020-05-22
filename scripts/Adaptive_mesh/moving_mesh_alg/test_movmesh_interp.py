#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:21:22 2020

@author: simo94


Rezoning approach for MMPDE and Physical PDE

r-adaptive method: Ceniceros's HOu

Compute the initial equidistributed by solving Winslow's equation 

div(M grad(u)) = 0 This can be seen as the steady-state solution of the hyperoblic PDE 

u_t = div(M grad(u)). This introduces relaxation in time and avoid mesh tangling 


Interpolate scalar field u from old to new mesh 

In a for loop:
    
    Integrate Physical PDE in time 
    
    Interpolate monitor function to computational domain 
    
    Solve equidistribution condition in the computational domain for the new mesh nodes
    
    Interpolate new solution to adapted mesh 

"""

# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

class Exact_sol(UserExpression):
    
    def __init__(self,x0,y0,r,t,w,sigma,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.r = r
        self.x0 = x0         # x-coord center of rotation 
        self.y0 = y0         # y-coord center of rotation 
        self.t = t           # time 
        self.w = w           # angular frequency
        self.sigma = sigma
    def eval(self, value, x):
            
        
        value[0] = exp(-self.sigma*((x[0] - (self.x0 + r*cos(self.w*self.t)))**2 
                   + (x[1] -  (self.y0 + r*sin(self.w*self.t)))**2))
        
    def value_shape(self):
        return ()

            
  
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


def transfer_function(fromFunc,toFunc):
 
    fromFunc.set_allow_extrapolation(True)
    
    # Create a transfer matrix from 2 different function spaces 
    A = PETScDMCollection.create_transfer_matrix(fromFunc.ufl_function_space(),toFunc.ufl_function_space())
    toFunc.vector()[:] = A*fromFunc.vector()


N = 50

mesh_c = UnitSquareMesh(N,N)     # Computational mesh 


mesh = UnitSquareMesh(N,N)       # Physical mesh 
mesh_ = UnitSquareMesh(N,N)      # Copy of mesh needed for interpolation 


V = FunctionSpace(mesh_c,'CG',1)  # Function space for mesh location 
W = FunctionSpace(mesh_c,'CG',2)  # Function space for monitor function w


U = FunctionSpace(mesh,'CG',3)  # Function space for solution u 

## Parameters exact solution 

x0 = 0.5
y0 = 0.5
t = 0
r = 0.1
w_freq = 2*pi
sigma = 80


# Assign initial condition for u
u0 = Function(U)
u0.interpolate(Exact_sol(x0,y0,r,t,w_freq,sigma,degree=5))


# Evaluate monitor function 
beta = 1.0
w0 = project(sqrt(1 + beta*inner(grad(u0),grad(u0))),U) 

w = Function(W)
w.interpolate(w0)


###############################################################################

## Solve Winslow's diffusion method until steady state 

# Impose Dirichlet and Neumann boundary conditions 
# At the first iteration x_old and y_old are computational mesh coordinates


# Dirichlet BC
# method = 'pointwise' and mesh = mesh_c  


x_D = Expression('x[0]',element = V.ufl_element())   
y_D = Expression('x[1]',element = V.ufl_element())

bc_x = DirichletBC(V,x_D,'on_boundary')
bc_y = DirichletBC(V,y_D,'on_boundary')


x_trial = TrialFunction(V)
y_trial = TrialFunction(V)

x_test = TestFunction(V)
y_test = TestFunction(V)


x_old = Function(V)
y_old = Function(V)

x_new = Function(V)
y_new = Function(V)


x_old.interpolate(Expression('x[0]',degree=2))
y_old.interpolate(Expression('x[1]',degree=2))

dx = Measure('dx',domain = mesh_c)

it = 0
max_iter = 10 # maximum number of iterations 
dt = 1e-3
dt_dif = 1e-4

file_mesh = File('Paraview_interp/mesh.pvd')
file_mesh << mesh,it

while it < max_iter:
    
    print(it)
    
    a = x_test*x_trial*dx + dt*inner(grad(x_test),w*grad(x_trial))*dx 
    L = x_test*x_old*dx   
    
    solve(a==L,x_new,bc_x)

    a = y_test*y_trial*dx + dt*inner(grad(y_test),w*grad(y_trial))*dx 

    L = y_test*y_old*dx 
    
    solve(a==L,y_new,bc_y)
    
    it +=1
    
    y_old = y_new
    x_old = x_new
    
    mesh_.coordinates()[:,0] = x_old.compute_vertex_values()
    mesh_.coordinates()[:,1] = y_old.compute_vertex_values()
    file_mesh << mesh_,it


# # x_test*w*inner(grad(x_trial),n)*ds
# a =  - inner(grad(x_test),w*grad(x_trial))*dx
# L = Constant(0.0)*x_test*dx
    
# solve(a==L,x_old,bc_x)
    
# #y_test*w*inner(grad(y_trial),n)*ds
# a = - inner(grad(y_test),w*grad(y_trial))*dx
# L = Constant(0.0)*y_test*dx
    
# solve(a==L,y_old,bc_y)

    
# mesh_.coordinates()[:,0] = x_old.compute_vertex_values()
# mesh_.coordinates()[:,1] = y_old.compute_vertex_values()


###############################################################################


#### interpolate u0 in new mesh  ####

U_ = FunctionSpace(mesh_,'CG',3)
u_ = Function(U_)


transfer_function(u0,u_)


mesh.coordinates()[:,0] = x_old.compute_vertex_values()
mesh.coordinates()[:,1] = y_old.compute_vertex_values()


u0.assign(u_)


# Build bounding box tree for cells of the mesh -> Expensive but needed 
# to avoid evaluatation at points outside the mesh 


mesh.bounding_box_tree().build(mesh)
mesh_.bounding_box_tree().build(mesh_)


u_exact = Function(U)
u_exact.interpolate(Exact_sol(x0,y0,r,t,w_freq,sigma,degree = 5))


print('Interpolation l2 rel error at time t',t,'is',errornorm(u_exact,u0))


###############################################################################


w_trial = TrialFunction(W)
w_test = TestFunction(W)

# Set up variables for advection equation 
bc_u = DirichletBC(U,Constant(0.0),'on_boundary')
u_trial = TrialFunction(U)
u_test = TestFunction(U)
u_new = Function(U)

## Select time step and method such that the local truncation error is less or 
# equal that the interpolation error 1e-6

# Take second order RK method with local truncation error dt^3

# time-step 1e-2 introduces too much diffusion 
beta = 1.0
dt = 1e-3
tf = 1.0
nt = int(tf/dt)


file_u = File('Paraview_interp/u.pvd')
file_u << u0,t

file_w = File('Paraview_interp/w.pvd')
file_w << w,t

err_vec = np.zeros(nt)
 

## Parameters for RK 2
# k1 = u0.copy(deepcopy = True)
# k2 = u0.copy(deepcopy = True)
# dx = Measure('dx',domain = mesh)
# a = u_test*u_trial*dx
# A = assemble(a)

for i in range(nt):

    print('iteration: ',i+1)
    
    
    # Solve Advection equation u_t + dot(c,grad(u)) == 0 
    
    # Use implicit/Explicit Euler scheme for 1nd order global truncation error  
    dx = Measure('dx',domain = mesh)
    
    c = Expression(('-w*r*sin(w*t)','w*r*cos(w*t)'),r=r,t = t,w = w_freq,degree=3)
    

    a = u_test*u_trial*dx + dt*u_test*inner(c,grad(u_trial))*dx 
    L = u_test*u0*dx 
    solve(a==L,u_new,bc_u)   
    
    #####################################
    ## Use Runge-Kutta 2 time integration 
    
    # L = - dt*u_test*inner(c,grad(u0))*dx 
    # b = assemble(L)
    
    # bc_u.apply(A)
    # solve(A,k1.vector(),b)
    
    # u_new.assign(u0+k1)
    
    # L = - dt*u_test*inner(c,grad(u_new))*dx 
    # b = assemble(L)
    # bc_u.apply(A)
    # solve(A,k2.vector(),b)
    
    # u_new.assign(u0 + 0.5*(k1 + k2))
    
    ###################################
    
    # Update monitor function w in physical mesh
    w0 = project(sqrt(1 + beta*inner(grad(u_new),grad(u_new))),U) 
    transfer_function(w0,w)
    
    ## Smooth w by diffusion  ##
    
    dx = Measure('dx',domain = mesh_c)
    a = (w_trial*w_test + dt_dif*inner(grad(w_trial), grad(w_test)))*dx
    L = w0*w_test*dx
    solve(a==L,w)

    gamma = np.max(w.vector()[:])
    #gamma = 1.0

    ## Compute the new mesh by solving MMPDE ## 
    
    dx = Measure('dx',domain = mesh_c)
    a = x_test*x_trial*dx + dt*inner(grad(x_test),w*grad(x_trial))*dx + dt*gamma*inner(grad(x_test),grad(x_trial))*dx  
    
    
    L = x_test*x_old*dx + dt*gamma*inner(grad(x_test),grad(x_old))*dx  
    
    solve(a==L,x_new,bc_x)

    a = y_test*y_trial*dx + dt*inner(grad(y_test),w*grad(y_trial))*dx + dt*gamma*inner(grad(y_test),grad(y_trial))*dx 

    L = y_test*y_old*dx + dt*gamma*inner(grad(y_test),grad(y_old))*dx 
    
    solve(a==L,y_new,bc_y)
    
    ## Interpolation ##
    
    mesh_.coordinates()[:,0] = x_new.compute_vertex_values()
    mesh_.coordinates()[:,1] = y_new.compute_vertex_values()

    
    transfer_function(u_new,u_)
    
    mesh.coordinates()[:,0] = x_new.compute_vertex_values()
    mesh.coordinates()[:,1] = y_new.compute_vertex_values()
    
    
    u0.assign(u_)
    
        
    # Update mesh coordinates    
    x_old = x_new
    y_old = y_new
    
    t += dt
    
    file_u << u0,t
    file_w << w,t
  
    
    u_exact.interpolate(Exact_sol(x0,y0,r,t,w_freq,sigma,element = U.ufl_element()))
    
    
    mesh.bounding_box_tree().build(mesh)
    mesh_.bounding_box_tree().build(mesh_)
    

    err_vec[i] = errornorm(u_exact,u0)    
    print('l2 error at time t',t,'is',errornorm(u_exact,u0))
    


plt.figure()
plt.plot(err_vec) 
plt.xlabel('iteration')
plt.ylabel('L2 error')
