#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:21:22 2020

@author: simo94
"""

# Clear all variables in workspace
from IPython import get_ipython;   
get_ipython().magic('reset -sf')


# Import modules
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np



def weighted_avg(v):
    
    '''
    Weighted averaging algorithm 
    
    Input: vector with n entries
    Output: smoothed vector with n entries
    
    '''
    n = np.len(v)
    w = np.zeros(n)
    
    for i in range(n):
        
        if i == 1:
           
            w[i] = 0.5*(v[0] + v[1])
        
        elif i == n-1:
            
            w[i] = 0.5*(v[i-1] + v[i])
            
        else:
            
            w[i] = 0.25*w[i-1] + 0.5*w[i] + 0.25*w[i+1]
           

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


N = 20

# mesh.coordinates has 51*51 rows and 2 columns 
mesh = UnitSquareMesh(N,N)
mesh_ = Mesh(mesh)
n = FacetNormal(mesh)


num_mesh_vertices = (N + 1) ** 2
mesh_vertex_coords = mesh.coordinates()

V = FunctionSpace(mesh,'CG',1)
U = FunctionSpace(mesh,'CG',2,constrained_domain=PeriodicBoundary())
W = FunctionSpace(mesh,'CG',1,constrained_domain=PeriodicBoundary())


u = Function(U)

# Assign initial condition for u in the physical space, which coincides with the computational one
u.interpolate(Expression('0.1*exp(-50.0*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)))',degree=5)) 
plot(u)


x_D = Expression('x[0]',degree=2)
y_D = Expression('x[1]',degree=2)

bc_x = DirichletBC(V,x_D,'on_boundary')
bc_y = DirichletBC(V,y_D,'on_boundary')

x_expr = Expression("x[0]",degree=3)
y_expr = Expression("x[1]",degree=3)

x_old = project(x_expr, V)
y_old = project(y_expr, V)

x_test = TestFunction(V)
y_test = TestFunction(V)

x = Function(V)
y = Function(V)


# The monitor function w is a function of grad(u) in the computational domain. 

beta = 5.0
w = Function(W)
w = project(sqrt(1 + beta*beta*(u.dx(0)*u.dx(0) + u.dx(1)*u.dx(1))),U)
#w.assign(w/np.sum(w.vector()[:]))
#w = weighted_avg(w)

'weak formulation of the equidistributing mesh'

# div(w grad(x)) == 0
# div(w grad(y)) == 0

#F = dt*x_test*div(w*grad(x_old))*dx 
F = 0
F = x_test*w*inner(grad(x),n)*ds - inner(grad(x_test),w*grad(x))*dx
jac = derivative(F,x)

# define Nonlinear variational problem with boundary conditions 
problem = NonlinearVariationalProblem(F,x,J = jac,bcs = bc_x)
solver = NonlinearVariationalSolver(problem)
        
# set solver paramters 
prm = solver.parameters
        
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['relative_tolerance'] = 1e-8
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['maximum_iterations'] = 100


solver.solve()
          
#F = dt*inner(y_test,div(w*grad(y_old)))*dx 

F = 0
F = y_test*w*inner(grad(y),n)*ds - inner(grad(y_test),w*grad(y))*dx

jac = derivative(F,y)

# define Nonlinear variational problem with boundary conditions 
problem = NonlinearVariationalProblem(F,y,J = jac,bcs = bc_y)
solver = NonlinearVariationalSolver(problem)
        
# set solver paramters 
prm = solver.parameters
        
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['relative_tolerance'] = 1e-8
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['maximum_iterations'] = 100
solver.solve()



mesh_.coordinates()[:,0] = x.compute_vertex_values()
mesh_.coordinates()[:,1] = y.compute_vertex_values()
   

plot(mesh_)

u_new = TrialFunction(U)
u_test = TestFunction(U)
u_old = u.copy()


x_trial = TrialFunction(V)
y_trial = TrialFunction(V)


dt = 0.01
dtau = 0.01

tf = 0.2

it = 0

beta = 5.0


nt = int(tf/dt)
diff = np.zeros(nt)

for i in range(nt):
    
    J = project(x.dx(0)*y.dx(1) - x.dx(1)*y.dx(0),V)
    
    u_x = (1/J)*((y.dx(1)*u).dx(0) - (y.dx(0)*u).dx(1))
    u_y = (1/J)*(-(x.dx(1)*u).dx(0) + (x.dx(0)*u).dx(1))
    
    #c = Expression(('-x[1]','x[0]'),degree=3)
    c = Expression(('1.0','0.0'),degree=3)
    
    
    # compute derivative x° and y° using x_old and x_new with artificial time step dtau
    
    x_t = project((x - x_old)/dt,V)
    y_t = project((y - y_old)/dt,V)
    
    a = u_test*u_new*dx 
    L =  u_test*u_old*dx  + dt*u_test*(u_x*x_t  + u_y*y_t - (c[0]*u_x + c[1]*u_y))*dx  
      
    solve(a==L,u)
    
    
    #plt.figure(i)
    #plot(u)
    u_old.assign(u)
   
    x_old.assign(x)
    y_old.assign(y)

    # Update monitor function w in computational mesh
    w = project(sqrt(1 + beta*beta*(u.dx(0)*u.dx(0) + u.dx(1)*u.dx(1))),W)
    
    # Smooth and normalize w 
    w.assign(w/np.sum(w.vector()[:]))
    

    gamma = dt*np.max(w.vector()[:])
    
    
    a = (x_test*x_trial)*dx - gamma*x_test*div(grad(x_trial))*dx

    #L = dt*x_test*div(w*grad(x_old))*dx 
    L =  dtau*x_test*div(w*grad(x_old))*dx +  x_test*x_old*dx - gamma*x_test*div(grad(x_old))*dx 

    solve(a==L,x,bc_x)


    a = (y_test*y_trial)*dx - gamma*y_test*div(grad(y_trial))*dx

    #L = dt*x_test*div(w*grad(x_old))*dx 
    L = dtau*y_test*div(w*grad(y_old))*dx +  y_test*y_old*dx - gamma*y_test*div(grad(y_old))*dx 


    solve(a==L,y,bc_y)
    
    diff_x = errornorm(x,x_old)/norm(x_old)
    diff_y = errornorm(y,y_old)/norm(y_old)
    diff[it] = diff_x + diff_y
    
    it +=1
    x_old.assign(x)
    y_old.assign(y)
    

    mesh_.coordinates()[:,0] = x.compute_vertex_values()
    mesh_.coordinates()[:,1] = y.compute_vertex_values()
   
    
    plt.figure(i)
    plot(mesh_)
    

    #a = x_test*x_trial*dx
    #L = dt*x_test*div(w*grad(x_old))*dx 
    #L = dtau*(x_test*w*inner(grad(x),n)*ds - inner(grad(x_test),w*grad(x_old))*dx) + x_test*x_old*dx

    #solve(a==L,x,bc_x)

    #a = y_test*y_trial*dx 
    #L = dt*inner(y_test,div(w*grad(y_old)))*dx 
    #L = dtau*(y_test*w*inner(grad(y),n)*ds - inner(grad(y_test),w*grad(y_old))*dx) + y_test*y_old*dx
    #solve(a==L,y,bc_y)
    
    #F = y_test*w*inner(grad(y),n)*ds - inner(grad(y_test),w*grad(y))*dx
    #jac = derivative(F,y)

    # define Nonlinear variational problem with boundary conditions 
    #problem = NonlinearVariationalProblem(F,y,J = jac,bcs = bc_y)
    #solver = NonlinearVariationalSolver(problem)
    #solver.solve()


    
        