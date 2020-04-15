#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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




def diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old):
     
   'Find vorticity and flux, then create form with unknown (u,h)'
        
   a = 0
   L = 0
    
   f = Constant(10.0)
    
   u_old,h_old = split(sol_old)


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
    
   solve(A,diag_old.vector(),b)
   vort_old,flux_old = diag_old.split()
    
   return vort_old,flux_old
    



def weak_form(sol_old,sol_test,vort_old,flux_old,S,dt):
    
   "weak form of Shallow Water equations in vector-invariant form"
    
   g = Constant(10.0)
   
   
   u_old,h_old = split(sol_old)               # Function
   u_test,h_test = split(sol_test)            # Test function  
   S1, S2, S3 = split(S)
     
   L = 0
    
   #### Momentum equation ####
       
    
   # Advection term 
   A_momentum = dt*(u_test[0]*vort_old*flux_old[1] 
                     - u_test[1]*vort_old*flux_old[0])*dx
    

   # Gradient term     
   C_momentum = dt*inner(div(u_test),g*h_old + 0.5*inner(u_old,u_old))*dx
   
   
   L += C_momentum + A_momentum + dt*(u_test[0]*S1 + u_test[1]*S2)*dx
    
    
   #### Continuity equation #### 

    
   rhs_continuity = -dt*(h_test*div(flux_old))*dx + dt*h_test*S3*dx
   L += rhs_continuity

    
   return L


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
    # divide by 1000
    #expr = Expression(('sin(4.0000000000*pi*x[1])/1000','0.00000000000',
     #               '10.000000000000 + 1.0000000/(4.0000000000*pi*1000)*cos(4.0000000000*pi*x[1])'),element = W1.ufl_element())
    
   
    #expr = Expression(('(1/1000.0)*sin(2*pi*x[0])*sin(x[1])','(1/1000.0)*2*pi*cos(2*pi*x[0])*cos(x[1])','10.0 + (1/1000.0)*sin(2*pi*x[0])*cos(x[1])'),element = W1.ufl_element())
    #expr = Expression(('0.0','sin(2*pi*x[0])','1.0 + (1/4*pi)*sin(4*pi*x[1])'),element = W1.ufl_element())
    expr = Expression(('sin(pi*x[1])','0.0000000','10.000 + sin(2*pi*x[0])*sin(pi*x[1])'),element = W1.ufl_element())
    
    
    sol_old.interpolate(expr)
    
    t = 0.0
    nt = int(tf/dt) 
    
    'Implementation of the Runge-Kutta method'
    
    sol_n = Function(W1)
    
    k1 = Function(W1)     
    k2 = Function(W1)
    k3 = Function(W1) 
    k4 = Function(W1)
   
    
    
    Z = FiniteElement('CG',mesh.ufl_cell(),2)
    W_elem = MixedElement([Z,Z,Z])
    Z = FunctionSpace(mesh,W_elem)
    
    
    S = Function(Z)
    
    S_exp = Expression(('10.0*2*pi*cos(2*pi*x[0])*sin(pi*x[1])',
                        '10.0*sin(pi*x[1]) + 10.0*sin(2*pi*x[0])*pi*cos(pi*x[1])',
                        '2*pi*cos(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])'),degree=5)
    
    S.interpolate(S_exp)
    
    scalars = np.zeros(4*(nt+1)).reshape(nt+1,4)
    devs_vec = np.zeros(3*(nt+1)).reshape(nt+1,3)
     
    it = 0
    
    
    u_0,h_0 = sol_old.split(deepcopy = True)
    q_0,flux_0  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
    
    
    # Assemble mass matrix once before starting iterations
    a = 0
    
    u_test,h_test = split(sol_test)  
    u,h = split(sol)
    
    lhs_momentum = inner(u_test,u)*dx
    a += lhs_momentum
    
    lhs_continuity = (h_test*h)*dx
    a += lhs_continuity
    
    A = assemble(a)
    
    
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

        sol_n = sol_old.copy(deepcopy = True)
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        L = weak_form(sol_old,sol_test,q_old,flux_old,S,dt)    
   
        b = assemble(L)
        solve(A,k1.vector(),b)
        sol_old.assign(sol_n + 0.5*k1)      
        
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        L = weak_form(sol_old,sol_test,q_old,flux_old,S,dt)    
   
        b = assemble(L)
        solve(A,k2.vector(),b)
        sol_old.assign(sol_n + 0.5*k2)
        
       
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        L = weak_form(sol_old,sol_test,q_old,flux_old,S,dt)    
   
        b = assemble(L)
        solve(A,k3.vector(),b)
        sol_old.assign(sol_n + k3)
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        L = weak_form(sol_old,sol_test,q_old,flux_old,S,dt)    
      

        b = assemble(L)
        solve(A,k4.vector(),b)
        sol_old.assign(sol_n + 1/6*(k1 + 2*k2 + 2*k3 + k4))
        
        u_f,h_f = sol_old.split(deepcopy = True)
        q_f,flux_f  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        
        
        dif_u = errornorm(u_0,u_f)
        dif_h = errornorm(h_0,h_f)
        dif_q = errornorm(q_0,q_f)
        
        devs_vec[it,:] = dif_u,dif_h,dif_q
        scalars[it,:] = diagnostics(q_f,u_f,flux_f,h_f)      
          
        # Compute CFL condition 
        max_u = max(u_0.vector()[:])
        cfl = (max_u)*dt/mesh.hmin()
        
        
        # Adapt the time step in order to have CFL condition equal to 0.9
        #print('time step ')
        #dt  = 0.1*mesh.hmin()/max_u 
    
        if cfl > 1.0:
            print('cfl condition is not satisfied')
            
        
        if dif_u/norm(u_0) < 1e-1 and dif_h/norm(h_0) < 1e-1:
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