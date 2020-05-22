#!/usr/bin/env python
# coding: utf-8

# Moving Mesh algorithm with triangular mesh using the method by Ceniceros and Hou
## Test error given by interpolation to new variables 



from fenics import *
import numpy as np
import matplotlib.pyplot as plt


N = 20

mesh = UnitSquareMesh(N,N)

# mesh space
V = FunctionSpace(mesh,'CG',1)

# Initial function
U = FunctionSpace(mesh,'CG',3)


## advection variable as function of the computational mesh, no need of interpolation
u0 = Function(U)


# assign initial condition for u in the physical space, which coincides with the computational one
u0.interpolate(Expression('exp(-50*((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5)))',degree=5))

plot(u0)


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

u = Function(U)
n = FacetNormal(mesh)


beta = 1.0
w = project(sqrt(1 + beta*beta*(u0.dx(0)*u0.dx(0) + u0.dx(1)*u0.dx(1))),U)

plot(w)

a = x_test*w*inner(grad(x_trial),n)*ds - inner(grad(x_test),w*grad(x_trial))*dx
L = Constant(0.0)*x_test*dx
    
solve(a==L,x,bc_x)
    
#a = y_test*div(w*grad(y_trial))*dx
a = y_test*w*inner(grad(y_trial),n)*ds - inner(grad(y_test),w*grad(y_trial))*dx
L = Constant(0.0)*y_test*dx
    
solve(a==L,y,bc_y)


mesh_ = Mesh(mesh)

mesh_.coordinates()[:,0] = x.compute_vertex_values()
mesh_.coordinates()[:,1] = y.compute_vertex_values()

plot(mesh_)


## Test error given by interpoaltion to new mesh 

# u0 still defined on old mesh 

## Create new function space and project 
U_proj = FunctionSpace(mesh_,'CG',3)

u_proj = Function(U_proj)

u_proj.interpolate(u0)


err = errornorm(u0,u_proj,norm_type='l2')        
print('err CG',err)

## Test error with Raviart-Thomas space 

RT1 = VectorFunctionSpace(mesh, "CG", 2)
RT2 = FunctionSpace(mesh_, "RT", 1)

v = Expression(('sin(10.0*x[0])*sin(10.0*x[1])','0.0'), degree=5)

v1 = Function(RT1)
v2 = Function(RT2)

v1.interpolate(v)
v2.interpolate(v1)

err = errornorm(u0,u_proj,norm_type='l2')        
print('err RT',err)



############################################################ 

mesh.coordinates()[:,0] = mesh_.coordinates()[:,0] 
mesh.coordinates()[:,1] = mesh_.coordinates()[:,0] 


u0.assign(u_proj)

f = File('u0.pvd')
f << u0


f2 = File('u_proj.pvd')
f2 << u_proj
#mesh.coordinates()[:,0] = mesh_.coordinates()[:,0]
#mesh.coordinates()[:,1] = mesh_.coordinates()[:,1]

