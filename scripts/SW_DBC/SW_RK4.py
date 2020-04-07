#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Created on Sun Feb 16 16:20:17 2020

@author: simo94

"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt




def diagnostics(vort,u,flux,h):

    g = Constant(10.0)  # gravity
    f = Constant(10.0)  # Coriolis term
    
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





def weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt,all_output=None):
    
    " Weak formulation of Shallow water system runge kutta method"
        
    a = 0
    L = 0
    
    g = Constant(10.0) 
    f = Constant(10.0)
    
    u_old,h_old = split(sol_old)               # Function t_n
    u,h = split(sol)                           # Function t_n+1
    u_test,h_test = split(sol_test)            # Test function  
     
    'Find vorticity and flux, then create form with unknown (u,h)'

    #### vorticity ####
    
    vort,flux = split(diag_trial)
    vort_test,flux_test = split(diag_test)
    vort_old,flux_old = split(diag_old)
    
    lhs_vort = inner(vort_test,vort*h_old)*dx
    rhs_vort = (vort_test.dx(1)*u_old[0] - vort_test.dx(0)*u_old[1])*dx + inner(vort_test,f)*dx
    
    #### flux  ####

    lhs_flux = inner(flux,flux_test)*dx
    rhs_flux = inner(h_old*u_old,flux_test)*dx
    
    
    a = lhs_flux + lhs_vort
    L = rhs_flux + rhs_vort
    
    A = assemble(a)
    b = assemble(L)
    
    
    per = Expression(('(10.0 - 4*pi*cos(4.0000000000*pi*x[1]))/(10.000000000000 + 1.0000000/(4.0000000000*pi)*cos(4.0000000000*pi*x[1]))',
                      'sin(4.0000000000*pi*x[1])*(10.000000000000 + 1.0000000/(4.0000000000*pi)*cos(4.0000000000*pi*x[1]))','0.0'),element = W2.ufl_element())
    
    bc2 = DirichletBC(W2,per, 'on_boundary')
    bc2.apply(A,b)
    
    solve(A,diag_old.vector(),b)
    vort_old,flux_old = diag_old.split()
    
    a = 0
    L = 0
    
    #### Momentum equation ####
       
    # mass matrix 
    lhs_momentum = inner(u_test,u)*dx
    a += lhs_momentum
    
    
    # Advection term 
    A_momentum = dt*(u_test[0]*vort_old*flux_old[1] 
                     - u_test[1]*vort_old*flux_old[0])*dx
    
    # APV stabilization
    #tau = dt/2
    #Q_term = dt*(u_test[1]*tau*(u_old[0]*(vort_old*flux_old[0]).dx(0) + u_old[1]*(vort_old*flux_old[0]).dx(1))
                               # - u_test[0]*tau*(u_old[0]*(vort_old*flux_old[1]).dx(0) + u_old[1]*(vort_old*flux_old[1]).dx(1)))*dx

    # Gradient term     
    C_momentum = dt*inner(div(u_test),g*h_old + 0.5*inner(u_old,u_old))*dx
    
    L += C_momentum + A_momentum + inner(u_test,u_old)*dx
    
    
    #### Continuity equation #### 

    lhs_continuity = (h_test*h)*dx
    a += lhs_continuity
    
    rhs_continuity = -dt*(h_test*div(flux_old))*dx + (h_test*h_old)*dx
    L += rhs_continuity

    
    if all_output:
        return a,L,vort_old,flux_old
    else:
        return a,L



def solver(mesh,W1,W2,dt,tf,output = None,lump = None):
    
    "Define the problem with initial condition"
    
    # (u,h): sol  (vort,flux): diag
    
    # Trial functions
    sol = TrialFunction(W1)
    diag_trial = TrialFunction(W2)
    
    # Test functions
    sol_test = TestFunction(W1)
    diag_test = TestFunction(W2)
    
    # Define initial conditions  
    sol_old = Function(W1)
    diag_old = Function(W2)
    
    
    # Set initial values for velocity and height field
    
    #f = Expression(('0.0','sin(2*pi*x[0])',
                   # '1 + 1/(4*pi)*sin(4*pi*x[1])'),element = W1.ufl_element())
    
    
    
    f = Expression(('sin(4.0000000000*pi*x[1])','0.00000000000',
                    '10.000000000000 + 1.0000000/(4.0000000000*pi)*cos(4.0000000000*pi*x[1])'),element = W1.ufl_element())
    
   
    sol_old.interpolate(f)
    
    t = 0.0
    nt = int(tf/dt) 
    
    'Implementation of the Runge-Kutta method'
    
    sol_n = Function(W1)
    sol_temp = Function(W1)
    
    k1 = Function(W1)     
    k2 = Function(W1)
    k3 = Function(W1) 
    k4 = Function(W1)
   
    solf = Function(W1)
    
    periodic = Expression(("sin(4.0000000000*pi*x[1])", "0.0",'10.000000 + 1.0000000/(4.0000000000*pi)*cos(4.0000000000*pi*x[1])'),element = W1.ufl_element())
    bc1 = DirichletBC(W1,periodic, 'on_boundary')

    scalars = np.zeros(4*(nt+1)).reshape(nt+1,4)
    error_vec = np.zeros(3*(nt+1)).reshape(nt+1,3)
     
    it = 0
    
    u_0,h_0 = sol_old.split(deepcopy = True)
    
    sol_paraw = Function(W1)
    u_paraw,h_paraw = sol_paraw.split(deepcopy = True)
    
    
    if output:
        print('Writing in pvd file')
        ufile = File('SW_paraview/sw_test_u.pvd')
        hfile = File('SW_paraview/sw_test_h.pvd')
        h_0.rename('h_0','h')
        u_0.rename('u_0','u')
        ufile << u_0,t
        hfile << h_0,t
    
    a,L,vort_0,flux_0 = weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt,all_output=True)    
    
    while(it <= nt):    

            
        sol_n = sol_old.copy(deepcopy = True)
        
        a,L = weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt)    
   
        A = assemble(a)
        b = assemble(L)
        bc1.apply(A,b)
        solve(A,sol_temp.vector(),b)
        k1.assign(sol_temp - sol_old)
        sol_old.assign(sol_n + 0.5*k1)      
        
        a,L = weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt)   

        A = assemble(a)        
        b = assemble(L)
        bc1.apply(A,b)
        solve(A,sol_temp.vector(),b)
        k2.assign(sol_temp - sol_old)
        sol_old.assign(sol_n + 0.5*k2)
        
            
        a,L = weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt)
        
        A = assemble(a)
        b = assemble(L)
        bc1.apply(A,b)
        solve(A,sol_temp.vector(),b)
        k3.assign(sol_temp - sol_old)
        sol_old.assign(sol_n + k3)
        
        
        a,L = weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt)   
        A = assemble(a)
        b = assemble(L)
        bc1.apply(A,b)
        solve(A,sol_temp.vector(),b)
        k4.assign(sol_temp - sol_old)
        sol_old.assign(sol_n + 1/6*(k1 + 2*k2 + 2*k3 + k4))
        
        u_f,h_f = sol_old.split(deepcopy = True)
    
        a,L,vort_f,flux_f = weak_form(W2,sol_old,sol_test,sol,diag_test,diag_trial,diag_old,dt,all_output=True)    
     
        
        dif_u = errornorm(u_0,u_f)
        dif_h = errornorm(h_0,h_f)
        dif_q = errornorm(vort_0,vort_f)
        
        #u_paraw.vector()[:] = (u_f.vector()[:] - u_0.vector()[:])
        #h_paraw.vector()[:] = (h_f.vector()[:] - h_0.vector()[:])
        
        
        
        error_vec[it,:] = dif_u,dif_h,dif_q
        scalars[it,:] = diagnostics(vort_f,u_f,flux_f,h_f)      
    
        if output:
            u_f.rename('u_f','u')
            h_f.rename('h_f','h')
            ufile << u_f,t
            hfile << h_f,t          

        if dif_u < 1e-1 and dif_h < 1e-1:
            # Move to next time step
            t += dt
            it = it + 1
        else:
            Warning('The RK scheme diverges')
            print('Solver diverges')
            return u_f,h_f,error_vec,scalars

    return u_f,h_f,vort_f,error_vec,scalars