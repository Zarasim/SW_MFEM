#!/usr/bin/env python3
# -*- coding: utf-8 -*-




"""
Created on Fri Apr 17 12:21:22 2020

@author: simo94


Solve Equation directly in computational domain and avoid interpolation 

"""

# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np


## Define User Expression Class for interpolating the monitor function 
class MyExpression(UserExpression):
    
    def __init__(self, beta,grad_f,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.beta = beta
        self.grad_f = grad_f
        
    def eval(self, value, x):
            
            value[0] =  sqrt( 1.0 + self.beta*(self.grad_f(x)[0]*self.grad_f(x)[0] + self.grad_f(x)[1]*self.grad_f(x)[1]))
            

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


N = 30

# constrained_domain=PeriodicBoundary()
mesh = UnitSquareMesh(N,N)
mesh_ = Mesh(mesh)          # new adapted mesh 

n_vertices = (N+1)*(N+1)

V = FunctionSpace(mesh,'CG',1)  # Function space for mesh location 
W = FunctionSpace(mesh,'CG',2) # Function space for monitor function w
U = FunctionSpace(mesh,'CG',3,constrained_domain=PeriodicBoundary())  # Space where solution u is defined 
U_grad = VectorFunctionSpace(mesh,'CG',2) 

#w0 = Function(W)
u0 = Function(U)


# Assign initial condition for u in the physical space
u0.interpolate(Expression('exp(-50*((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5)))',element = U.ufl_element())) 
grad_u0 = project(grad(u0),U_grad)

plt.figure()


beta = 1.0
#expr =  MyExpression(beta = 1.0,grad_f = grad_u0,element = W.ufl_element())
#w0.interpolate(expr)

w0 = sqrt(1.0 + beta*(u0.dx(0)*u0.dx(0) + u0.dx(1)*u0.dx(1)))

#### Solve Winslow's equation 

## Assign Dirichlet BC defining User expression 
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

n = FacetNormal(mesh)


# x_test*w*inner(grad(x_trial),n)*ds 
a = - inner(grad(x_test),w0*grad(x_trial))*dx
L = Constant(0.0)*x_test*dx
    
solve(a==L,x_old,bc_x)
    

# y_test*w*inner(grad(y_trial),n)*ds
a = - inner(grad(y_test),w0*grad(y_trial))*dx
L = Constant(0.0)*y_test*dx
    
solve(a==L,y_old,bc_y)
 

mesh_.coordinates()[:,0] = x_old.compute_vertex_values()
mesh_.coordinates()[:,1] = y_old.compute_vertex_values()


###############################################################################


file_u = File('u.pvd')
file_u << u0


file_w = File('w.pvd')
#file_w << w0


file_mesh = File('mesh.pvd')
file_mesh << mesh_


u_trial = TrialFunction(U)
u_test = TestFunction(U)
u_new = Function(U)


w_trial = TrialFunction(W)
w_test = TestFunction(W)
w = Function(W)

t = 0


beta = 0.7
dt = 5e-4
dt_dif = 1e-4
tf = 1.0

nt = int(tf/dt)
it = 0


c = Expression(('1.0','0.0'),degree=2)
    

for i in range(nt):
    
    print('iteration: ',i+1)

    ## compute the monitor function w
    #expr =  MyExpression(beta = 1.0,grad_f = grad_u0,element = W.ufl_element())
    #w0.interpolate(expr)    
    w0 = sqrt(1.0 + beta*inner(grad(u0),grad(u0)))
    
    ## Smooth w by solving simple diffusion 
    a = (w_trial*w_test + dt_dif*inner(grad(w_trial), grad(w_test)))*dx
    L = w0*w_test*dx
    solve(a==L,w)

    gamma = np.max(w.vector()[:])

    ## Compute the new mesh by solving the Ceniceros MMPDE 
    # Use implicit Euler time-step + diffusion 
    
    a = x_test*x_trial*dx + dt*gamma*inner(grad(x_test),grad(x_trial))*dx + dt*inner(grad(x_test),w*grad(x_trial))*dx
    L = x_test*x_old*dx  + dt*gamma*inner(grad(x_test),grad(x_old))*dx 
    
    solve(a==L,x_new,bc_x)

    a = y_test*y_trial*dx + dt*gamma*inner(grad(y_test),grad(y_trial))*dx + dt*inner(grad(y_test),w*grad(y_trial))*dx 

    L = y_test*y_old*dx + dt*gamma*inner(grad(y_test),grad(y_old))*dx 
    
    solve(a==L,y_new,bc_y)
         
    
    mesh_.coordinates()[:,0] = x_new.compute_vertex_values()
    mesh_.coordinates()[:,1] = y_new.compute_vertex_values()
    
    
    #### Update physical solution using the new mesh and the mesh velocity
    
    # Advection equation: u_t - u_x*x_t - u_y*y_t + c*u_x = 0
    
    #  u_test*((y_new.dx(1)*u_trial).dx(0) - (y_new.dx(0)*u_trial).dx(1))/(x_new.dx(0)*y_new.dx(1) - x_new.dx(1)*y_new.dx(0))*(x_new - x_old)/dt*dx
    
    
    ## Error in advection equation at first time step 
    # Define Jacobian transformation
    J = x_new.dx(0)*y_new.dx(1) - x_new.dx(1)*y_new.dx(0)
    
    # Estimate mesh speed
    
    x_t = (x_new - x_old)/dt
    y_t = (y_new - y_old)/dt

    # Compute new spatial derivatives
    #u_x = (1/J)*((y_old.dx(1)*u0).dx(0) - (y_old.dx(0)*u0).dx(1))
    #u_y = (1/J)*(-(x_old.dx(1)*u0).dx(0) + (x_old.dx(0)*u0).dx(1))
    
    u_x = (1/J)*((y_new.dx(1)*u0).dx(0) - (y_new.dx(0)*u0).dx(1))
    u_y = (1/J)*(-(x_new.dx(1)*u0).dx(0) + (x_new.dx(0)*u0).dx(1))
    
    #u_test_x = (1/J)*((y_old.dx(1)*u_test).dx(0) - (y_old.dx(0)*u_test).dx(1))
    #u_test_y = (1/J)*(-(x_old.dx(1)*u_test).dx(0) + (x_old.dx(0)*u_test).dx(1))
    
    
    a = (u_test*u_trial/dt)*dx
    L = (u_test*u0/dt)*dx + u_test*u_x*x_t*dx + u_test*u_y*y_t*dx - u_test*(c[0]*u_x + c[1]*u_y)*dx
    
    #+ (c[0]*u_test_x + c[1]*u_test_y)*u0*dx - u_test*u0*inner(c,n)*ds
       
    solve(a==L,u_new)
     
    u0.assign(u_new)
    grad_u0 = project(grad(u0),U_grad)
    
    x_old = x_new
    y_old = y_new

    t += dt
    
    file_w << w,t
    file_u << u0,t
    file_mesh << mesh_,t