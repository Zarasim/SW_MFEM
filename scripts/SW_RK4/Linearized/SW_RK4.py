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

def diagnostics(u,h,S,dt,t):

    g = Constant(10.0)  # gravity
    
    S1,S2,S3 = split(S)
    
    # Energy
    E_kin = assemble(h*(u[0]*u[0] + u[1]*u[1])*dx)
    E_pot = assemble(g*h*h*dx)
    
    # Mass
    M = assemble(h*dx)
    
    #E =  assemble(0.5*(h*(u[0]*u[0] + u[1]*u[1]) + g*h*h)*dx)
    E = E_kin + E_pot
    
    return E_pot,E_kin,M,E


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


    


def weak_form(sol_old,sol_test,S,dt):
    
   "weak form of Shallow Water equations in vector-invariant form"
    
   g = Constant(10.0)
   f = Constant(10.0)
   #0.01
   #50.0*2*pi
   H = Constant(10.0)
   
   u_old,h_old = split(sol_old)               # Function
   u_test,h_test = split(sol_test)            # Test function  
   S1, S2, S3 = split(S)
     
   L = 0
    
   #### Momentum equation ####
       
    
   # Advection term 
   A_momentum = dt*f*(u_test[0]*u_old[1] -u_test[1]*u_old[0])*dx
    

   # Gradient term     
   C_momentum = dt*g*div(u_test)*h_old*dx
   
   
   # Source term 
   #S_momentum = dt*(u_test[0]*S1 + u_test[1]*S2)*dx
   
   
   # Anticipated potential vorticity
   # This removes inertia-gravity waves so that we can apply our moving mesh
   # strategy
   
   #tau = dt/2.0
   # 1st component of Q
   #Q_momentum =  + tau*(u_old[0]*(vort_old.dx(0)*flux_old[0] + vort_old*flux_old[0].dx(0)) \
   #                    + u_old[1]*(vort_old.dx(1)*flux_old[0] + vort_old*flux_old[0].dx(1)))*u_test[1]*dx \
   #             - tau*(u_old[0]*(vort_old.dx(0)*flux_old[1] + vort_old*flux_old[1].dx(0)) \
   #                    + u_old[1]*(vort_old.dx(1)*flux_old[1] + vort_old*flux_old[1].dx(1)))*u_test[0]*dx
   
   
   L += C_momentum + A_momentum  #+ Q_momentum  #S_momentum
    
   #### Continuity equation #### 

    
   rhs_continuity = -dt*H*h_test*div(u_old)*dx
   L += rhs_continuity

    
   return L


def solver(mesh,W,u_0,h_0,dt,tf,output,lump,case):
    
    "Define the problem with initial condition"
    
    # (u,h): sol  (vort,flux): diag
 
   
    # Trial functions
    sol = TrialFunction(W)
   
    # Test functions
    sol_test = TestFunction(W)
    
    # Define initial conditions  
    sol_old = Function(W)
    
    
   
    
    ## Source term 
    S_elem = FiniteElement('CG',mesh.ufl_cell(),4)
    W_elem = MixedElement([S_elem,S_elem,S_elem])
    Z = FunctionSpace(mesh,W_elem)
    S = Function(Z)   
           
    
    ## 1D case
    
    if case == '1D':
        
        print('1D')
        expr = Expression(('sin(2.00000*pi*x[1])/1000','0.0000000',
                       '10.0000 + 1.0000/(2.00000*pi*1000)*cos(2.0000*pi*x[1])'),element = W.ufl_element())
        
        
        # expr_u = Expression(('sin(4.0*pi*x[1])','0.0'),element = CG_u.ufl_element())
        # expr_h = Expression('10.0 + 1.0/(4.0*pi)*cos(4.0*pi*x[1])',element = CG_h.ufl_element())
        
        
   
        # Source term for 1D case 
        S_exp = Expression(('0.0','0.0','0.0'),degree=4)  
    
   
    else:
        
        expr = Expression(('sin(pi*x[1])','0.0000000','10.000 + sin(2*pi*x[0])*sin(2*pi*x[1])'),element = W.ufl_element())
    
     
        # expr_u = Expression(('sin(pi*x[1])','0.00000'),element = CG_u.ufl_element())
        # expr_h = Expression('10.000 + sin(2*pi*x[0])*sin(pi*x[1])',element = CG_h.ufl_element())
        
         # Source term for 2D case on the right side
        S_exp = Expression(('10.0*2*pi*cos(2*pi*x[0])*sin(2*pi*x[1])',
                        '10.0*sin(pi*x[1]) + 10.0*sin(2*pi*x[0])*2*pi*cos(2*pi*x[1])',
                        '2*pi*cos(2*pi*x[0])*sin(pi*x[1])*sin(2*pi*x[1])'),degree=4)
   
    
    sol_old.interpolate(expr)
    S.interpolate(S_exp)

    
    t = 0.0
    nt = int(tf/dt) 
    it = 0 
    
    'Implementation of the Runge-Kutta method'
    
    sol_n = Function(W)
    
    k1 = Function(W)     
    k2 = Function(W)
    k3 = Function(W) 
    k4 = Function(W)
   
    
    scalars = np.zeros(4*(nt+1)).reshape(nt+1,4)
    devs_vec = np.zeros(2*(nt+1)).reshape(nt+1,2) 
    sol_vec = np.zeros(3*(nt+1)).reshape(nt+1,3) 
    
    # Assemble mass matrix once before starting iterations
    a = 0
    
    u_test,h_test = split(sol_test)  
    u,h = split(sol)
    
    lhs_momentum = inner(u_test,u)*dx
    a += lhs_momentum
    
    lhs_continuity = (h_test*h)*dx
    a += lhs_continuity
    
    A = assemble(a)
    
    Proj_space = FunctionSpace(mesh,'CG',2)
    
    if output:
        print('Writing in pvd file')
        ufile = File('SW_paraview/sw_test_u.pvd')
        hfile = File('SW_paraview/sw_test_h.pvd')
        u_0.rename('u_0','u')
        h_0.rename('h_0','h')
        ufile << u_0,t
        hfile << h_0,t
          
        
    while(it <= nt):    

        sol_n = sol_old.copy(deepcopy = True)

        
        L = weak_form(sol_old,sol_test,S,dt)    
   
        b = assemble(L)
        solve(A,k1.vector(),b)
        sol_old.assign(sol_n + 0.5*k1)      
        
   
        L = weak_form(sol_old,sol_test,S,dt)       
        
        b = assemble(L)
        solve(A,k2.vector(),b)
        sol_old.assign(sol_n + 0.5*k2)
        
       
        L = weak_form(sol_old,sol_test,S,dt)  
        
        b = assemble(L)
        solve(A,k3.vector(),b)
        sol_old.assign(sol_n + k3)
        

        L = weak_form(sol_old,sol_test,S,dt)  
        b = assemble(L)
        solve(A,k4.vector(),b)
        sol_old.assign(sol_n + 1/6*(k1 + 2*k2 + 2*k3 + k4))
        
        u_f,h_f = sol_old.split(deepcopy = True)
        
        
        ## Look at the oscillations in one single point over time for u,v and h
        # separately
            
        dif_u = errornorm(u_0,u_f,norm_type='l2', degree_rise=3)
        dif_h = errornorm(h_0,h_f,norm_type='l2', degree_rise=3)      
        
        devs_vec[it,:] = dif_u,dif_h
        
        
        ## For u defined in RT/BDM cannot extract directly components
        # project in piecewise linear function 
        u_fixed = project(u_f[0],Proj_space).vector()[30]
        v_fixed = project(u_f[1],Proj_space).vector()[30]
        h_fixed = h_f.vector()[30]
        
        sol_vec[it,:] = u_fixed,v_fixed,h_fixed  
        
        t += dt
        scalars[it,:] = diagnostics(u_f,h_f,S,dt,t)      
          
        # Compute CFL condition by looking at max(u_vector/h) for each cell
        # of the mesh 
        max_u = max(u_f.vector()[:])
        cfl = (max_u)*dt/mesh.hmin()
        
        if cfl > 1.0:
            print('cfl condition is not satisfied')
            
        
        # Move to next time step
        if dif_u > 10.0 or dif_h > 10.0: 
            
            Warning('The RK scheme diverges')
            print('Solver diverges')
            return 
        
        it = it + 1
        
        if output:
            u_f.rename('u_f','u')
            h_f.rename('h_f','h')
            ufile << u_f,t
            hfile << h_f,t  
            

    return sol_vec,devs_vec,scalars