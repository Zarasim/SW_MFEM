#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:29:02 2020

Shallow-Water equations solved using midpoint implicit method

This is used for solving solving the system Delaunay triangulation 


@author: simo94
"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

form_compiler_parameters = {"quadrature_degree": 6}


def diagnostics(vort,u,flux,h,vort_0,u_0,flux_0,h_0):

    g = Constant(10.0)  # gravity
    
    u = (u + u_0)/2.0
    h = (h + h_0)/2.0
    
    # Energy
    E = assemble(0.5*(h*inner(u,u) + g*h**2)*dx)
               
    # Enstrophy    
    Ens = assemble((vort**2)*h*dx)
    
    # Absolute vorticity 
    Absv = assemble(vort*h*dx)
        
    # Mass
    M = assemble(h*dx)

    return E,Ens,Absv,M



def lumping(a):
    
    'lumping the mass matrix only works for Lagrangian Elements'
    
    act = Constant((1.0,1.0,1.0))
    mass_action_form = action(a,act)

    # create vector with integral of u_j phi_j for each j
    diag_lumped = assemble(mass_action_form)

    A = assemble(a)
    # set all entries of vector to 0. Do it after assemble 
    A.zero()

    # assemble the mass action form and set them as diagonal elements of M_lumped
    A.set_diagonal(diag_lumped)
    
    return A



def weak_form(u_test,u_old,u,flux_test,flux,vort_test,vort_old,vort,h_test,
              h_old,h,theta,dt):
    
    " Weak formulation of Shallow water system "
    
    F = 0
    
    g = Constant(10.0) 
    f = Constant(50*2*pi)
    
    # weight u and h by theta to obtain a time-stepping method
    u_mid = (1.0-theta)*u_old + theta*u
    h_mid = (1.0-theta)*h_old + theta*h

    #### vorticity ####
    
    a = inner(vort_test,vort*h_mid)*dx
    L = (vort_test.dx(1)*u_mid[0] - vort_test.dx(0)*u_mid[1])*dx + f*vort_test*dx
       
    F += a - L
    
    #### flux  ####
    a = inner(flux,flux_test)*dx
    L = inner(h_mid*u_mid,flux_test)*dx
    
    F += a - L

    #### Momentum equation ####
    
    # Mass term 
    M_momentum = (inner(u_test,u) - inner(u_test,u_old))*dx
    F += M_momentum
    
    # Advection term 
    A_momentum = dt*(-u_test[0]*vort*flux[1] 
                     + u_test[1]*vort*flux[0])*dx
    F += A_momentum

    # Gradient term     
    C_momentum = -dt*inner(div(u_test),g*h_mid + 0.5*inner(u_mid,u_mid))*dx
    
    F += C_momentum
    
    #### Continuity equation #### 
    
    M_continuity = (h_test*h - h_test*h_old)*dx
    F += M_continuity

    Ct_continuity = dt*(h_test*div(flux))*dx

    F += Ct_continuity
    
    return F




def solver(mesh,W,dt,tf,output = None,lump = None):
    
    "Define the problem with initial condition"
    
    # (u,h): sol  (vort,flux): diag
    
    sol = Function(W)
    sol_old = Function(W)
    vort,u,flux,h = split(sol)

    # Test functions
    vort_test,u_test,flux_test,h_test = TestFunctions(W)


    # Define initial conditions
    expr = Expression(('0.0',
                    'sin(4.0*pi*x[1])/10','0.0',
                    '0.0','0.0',
                    '10.0 + (1/(4.0*pi*10))*cos(4.0*pi*x[1])'),element = W.ufl_element())
    
    sol_old.interpolate(expr)

       
    CG_u = VectorFunctionSpace(mesh,'CG',2)
    CG_h = FunctionSpace(mesh,'CG',2)
    u_0 = Function(CG_u)
    h_0 = Function(CG_h)
    expr_u = Expression(('sin(4.0*pi*x[1])/10','0.0'),element = CG_u.ufl_element())
    expr_h = Expression('10.0 + (1/(4.0*pi*10))*cos(4.0*pi*x[1])',element = CG_h.ufl_element())
    
    
    ## 2D case
    #expr = Expression(('sin(pi*x[1])','0.0000000','10.000 + sin(2*pi*x[0])*sin(pi*x[1])'),element = W1.ufl_element())     
    #expr_u = Expression(('sin(pi*x[1])','0.00000'),element = CG_u.ufl_element())
    #expr_h = Expression('10.000 + sin(2*pi*x[0])*sin(pi*x[1])',element = CG_h.ufl_element())
   
    
    u_0.interpolate(expr_u)
    h_0.interpolate(expr_h)
        

    # Assign initial conditions to trial functions
    sol.assign(sol_old) 
    
    # trapezium implicit time-step method
    theta = 0.5
    
    vort_old,u_old,flux_old,h_old = split(sol_old)
    
    # Pass separately each component of sol vector
    F = weak_form(u_test,u_old,u,flux_test,flux,vort_test,vort_old,vort,h_test,
                  h_old,h,theta,dt)    
   
    t = 0.0
    nt = int(tf/dt) 
    it = 0
            
    scalars = np.zeros(4*(nt+1)).reshape(nt+1,4)
    devs_vec = np.zeros(2*(nt+1)).reshape(nt+1,2)
 
    if output:
        print('Writing in pvd file')
        ufile = File('SW_paraview/sw_test_u.pvd')
        hfile = File('SW_paraview/sw_test_h.pvd')
        h_0.rename('h_0','h')
        u_0.rename('u_0','u')
        ufile << u_0,t
        hfile << h_0,t

           
    while(it <= nt):    

        
        
        # compute Jacobian of Nonlinear form F
        jac = derivative(F, sol)
        
        # define Nonlinear variational problem with boundary conditions 
        problem = NonlinearVariationalProblem(F,sol,J = jac)
        solver = NonlinearVariationalSolver(problem)
        
        # set solver paramters 
        prm = solver.parameters
        
        prm['newton_solver']['absolute_tolerance'] = 1e-12
        prm['newton_solver']['relative_tolerance'] = 1e-12
        prm['newton_solver']['linear_solver'] = 'lu'
        prm['newton_solver']['maximum_iterations'] = 10
        
        
        vort_old,u_old,f_old,h_old = sol_old.split(deepcopy = True)
        
        solver.solve()
        sol_old.assign(sol)      

        vort_f,u_f,flux_f,h_f = sol.split()
              
        scalars[it,:] = diagnostics(vort_f,u_f,flux_f,h_f,vort_old,u_old,f_old,h_old)
    
        dif_u = errornorm(u_0,u_f)
        dif_h = errornorm(h_0,h_f)
        
        devs_vec[it,:] = dif_u,dif_h        
            
        #if dif_u < 1e-1 and dif_h < 1e-1:
            # Move to next time step
        t += dt
        it = it + 1
        #else:
        #    Warning('The RK scheme diverges')
        #    print('Solver diverges')
        #    return 
        
        
        if output:
            u_f.rename('u_f','u')
            h_f.rename('h_f','h')
            ufile << u_f,t
            hfile << h_f,t


    return devs_vec,scalars















