#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:29:02 2020

Shallow-Water equations solved using midpoint implicit method

This is used for solving solving the system Delaunay triangulation 


@author: simo94
"""



"""

Created on Sun Feb 16 16:20:17 2020

@author: simo94

"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

#form_compiler_parameters = {"quadrature_degree": 6}



def diagnostics(vort,u,flux,h):

    g = Constant(10.0)  # gravity
  
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
    
    g = Constant(5.0) 
    f = Constant(5.0)
    
    # weight u and h by theta to obtain a time-stepping method
    u_mid = (1.0-theta)*u_old + theta*u
    h_mid = (1.0-theta)*h_old + theta*h
    vort_mid = (1.0-theta)*vort_old + theta*vort
    
    #### vorticity ####
    
    a = (inner(vort_test,vort*h) - inner(vort_test,vort_old*h_old))*dx
    L = dt*(vort_test.dx(0)*vort_mid*flux[0] + 
            vort_test.dx(1)*vort_mid*flux[1])*dx
    
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
    A_momentum = dt*(-u_test[0]*vort_mid*flux[1] 
                     + u_test[1]*vort_mid*flux[0])*dx
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
    vort,u,flux,h = split(sol)

    # Test functions
    vort_test,u_test,flux_test,h_test = TestFunctions(W)


    # Set initial values for velocity and height field 
    # divide by 1000
    #expr = Expression(('sin(4.0000000000*pi*x[1])/1000.0','0.00000000000',
    #'10.000000000000 + 1.0000000/(4.0000000000*pi*1000.0)*cos(4.0000000000*pi*x[1])'),element = W1.ufl_element())
    
    #expr = Expression(('(1/1000.0)*sin(2*pi*x[0])*sin(x[1])','(1/1000.0)*2*pi*cos(2*pi*x[0])*cos(x[1])','10.0 + (1/1000.0)*sin(2*pi*x[0])*cos(x[1])'),element = W1.ufl_element())
    #expr = Expression(('0.0','sin(2*pi*x[0])','1.0 + (1/4*pi)*sin(4*pi*x[1])'),element = W1.ufl_element())
    

    # Define initial conditions
    sol_old = Function(W)
    expr = Expression(('(2*pi*cos(2*pi*x[0]) + 5)/(1.0 + (1/(4*pi))*sin(4*pi*x[1]))',
                    '0.0','sin(2*pi*x[0])','0.0','0.0',
                    '1.0 + (1/(4*pi))*sin(4*pi*x[1])'),element = W.ufl_element())
    
    sol_old.interpolate(f)

    # zero initial conditions 
    vort_old,u_old,f_old, h_old = split(sol_old)

    # Assign initial conditions to trial functions
    sol.assign(sol_old) 
    
    # trapezium implicit time-step method
    theta = 0.5
    
    F = weak_form(u_test,u_old,u,flux_test,flux,vort_test,vort_old,vort,h_test,
                  h_old,h,theta,dt)    
    u,h,diagn = iter_solver(F,sol_old,sol,dt)
    
    
    t = 0.0
    nt = int(tf/dt) 
    
    'Implementation of the Runge-Kutta method'
    
    sol_n = Function(W1)
    
    it = 0
    
    # Assemble mass matrix once before starting iterations
    a = 0
    
    
    if output:
        print('Writing in pvd file')
        ufile = File('SW_paraview/sw_test_u.pvd')
        hfile = File('SW_paraview/sw_test_h.pvd')
        qfile = File('SW_paraview/sw_test_q.pvd')
        h_0.rename('h_0','h')
        u_0.rename('u_0','u')
        q_0.rename('q_0','q')
        ufile << u_0,t
        hfile << h_0,t
        qfile << q_0,t
      
            
    while(it <= nt):    

        
        
        # compute Jacobian of Nonlinear form F
        jac = derivative(F, sol)
        
        # define Nonlinear variational problem with boundary conditions 
        problem = NonlinearVariationalProblem(F,sol,J = jac)
        solver = NonlinearVariationalSolver(problem)
        
        # set solver paramters 
        prm = solver.parameters
        
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-8
        prm['newton_solver']['linear_solver'] = 'lu'
        prm['newton_solver']['maximum_iterations'] = 1000
        
        
        vort_0,u_0,f_0,h_0 = split(sol_old)
        
        solver.solve()
        sol_old.assign(sol)      

        vort,u_f,flux,h_f = split(sol)
              
        scalars[it,:] = diagnostics(vort,u,flux,h,vort_0,u_0,f_0,h_0)
    
        dif_u = errornorm(u_0,u_f)
        dif_h = errornorm(h_0,h_f)
        dif_q = errornorm(q_0,q_f)
        
        devs_vec[it,:] = dif_u,dif_h,dif_q
        
            
        if dif_u < 1e-1 and dif_h < 1e-1:
            # Move to next time step
            t += dt
            it = it + 1
        else:
            Warning('The RK scheme diverges')
            print('Solver diverges')
            return 
        
        
        if output:
            u_f.rename('u_f','u')
            h_f.rename('h_f','h')
            q_f.rename('q_f','q')
            ufile << u_f,t
            hfile << h_f,t  
            qfile << q_f,t          


    return devs_vec,scalars