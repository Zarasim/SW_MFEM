#!/usr/bin/env python
# coding: utf-8

# ## Winslow's diffusion relaxation method




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
U_grad = VectorFunctionSpace(mesh,'CG',2) # Function space for grad(u) projection 
W_grad = VectorFunctionSpace(mesh_c,'CG',2) # Function space for grad(u) evaluation in computational domain 

## Parameters exact solution 

x0 = 0.7
y0 = 0.5
t = 0
r = 0.1
w_freq = 2*pi


u_exact = Function(U)
w = Function(W)
u0 = Function(U)


# Assign initial condition for u
u0.interpolate(Expression('exp(-50*((x[0] - 0.7)*(x[0] - 0.7) + (x[1] - 0.5)*(x[1] - 0.5)))',degree=5)) 
#u0.interpolate(Exact_sol(x0,y0,r,t,w_freq))


u_exact.interpolate(Exact_sol(x0,y0,r,t,w_freq))


beta = 1.0
w0 = project(sqrt(1 + beta*inner(grad(u0),grad(u0))),U) 
w.interpolate(w0)

plot(w)

### Solve Winslow's diffusion method until steady state 

# Impose Dirichlet and Neumann boundary conditions 
# At the first iteration x_old and y_old are computational mesh coordinates


# Dirichlet BC
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

rel_tol = 1e-15  # set relative tolerance 
err = 1

it = 0
max_iter = 10 # maximum number of iterations 
dt = 1e-3

## Initial mesh is given as solution of Winslow's equation without term dt 


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
    
    print(errornorm(x_old,x_new))
    
    it +=1
    
    y_old = y_new
    x_old = x_new
    
    mesh.coordinates()[:,0] = x_old.compute_vertex_values()
    mesh.coordinates()[:,1] = y_old.compute_vertex_values()
    file_mesh << mesh,it
