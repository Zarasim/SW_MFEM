#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Created on Sun Feb 16 16:20:17 2020

@author: simo94

"""


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


class MyExpression(UserExpression):
    
    def __init__(self, sigma,xa,xb,yc,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.sigma = sigma
        self.xa = xa
        self.xb = xb
        self.ya = yc
        
    def eval(self, value, x):
            dxa = x[0] - self.xa
            dxb = x[0] - self.xb
            
            dy = x[1] - self.ya
            
            value[0] =  -dy*(exp(-(dxa*dxa + dy*dy)/(2.0*self.sigma)) + exp(-(dxb*dxb + dy*dy)/(2.0*self.sigma)))/(2.0*pi*self.sigma**2)
            value[1] = (dxa*exp(-(dxa*dxa + dy*dy)/(2.0*self.sigma)) + dxb*exp(-(dxb*dxb + dy*dy)/(2.0*self.sigma)))/(2.0*pi*self.sigma**2)
            value[2] = (1.0/(2.0*pi*self.sigma))*(exp(-(dxa*dxa + dy*dy)/(2.0*self.sigma)) +  exp(-(dxb*dxb + dy*dy)/(2.0*self.sigma)))

            
    def value_shape(self):
        return (3,)



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



def diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old):
     
   'Find vorticity and flux, then create form with unknown (u,h)'
        
   a = 0
   L = 0
    
   g = Constant(10.0) 
   f = Constant(10.0)
    
   u_old,h_old = split(sol_old)               # Function t_n
   u,h = split(sol)                           # Function t_n+1
   u_test,h_test = split(sol_test)            # Test function  
     

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
    



def weak_form(sol_old,sol_test,sol,vort_old,flux_old,dt):
    
    "weak form of Shallow Water equations in vector-invariant form"
    
    a = 0
    L = 0
    
    #### Momentum equation ####
       
    # mass matrix 
    lhs_momentum = inner(u_test,u)*dx
    a += lhs_momentum
    
    
    # Advection term 
    A_momentum = dt*(u_test[0]*vort_old*flux_old[1] 
                     - u_test[1]*vort_old*flux_old[0])*dx
    

    # Gradient term     
    C_momentum = dt*inner(div(u_test),g*h_old + 0.5*inner(u_old,u_old))*dx
    
    L += C_momentum + A_momentum 
    
    
    #### Continuity equation #### 

    lhs_continuity = (h_test*h)*dx
    a += lhs_continuity
    
    rhs_continuity = -dt*(h_test*div(flux_old))*dx
    L += rhs_continuity

    
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
    expr = MyExpression(sigma = 0.005,xa = 0.4 ,xb = 0.6,yc = 0.5,element = W1.ufl_element())
   
    sol_old.interpolate(expr)
    
    t = 0.0
    nt = int(tf/dt) 
    
    'Implementation of the Runge-Kutta method'
    
    sol_n = Function(W1)
    
    k1 = Function(W1)     
    k2 = Function(W1)
    k3 = Function(W1) 
    k4 = Function(W1)
   
    
    scalars = np.zeros(4*(nt+1)).reshape(nt+1,4)
    error_vec = np.zeros(3*(nt+1)).reshape(nt+1,3)
     
    it = 0
    

    q_0,flux_0  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
    u_0,h_0 = sol_old.split(deepcopy = True)

        
    if output:
        print('Writing in pvd file')
        qfile = File('SW_paraview_vortex/sw_test_q.pvd')

    
    while(it <= nt):    

            
        sol_n = sol_old.copy(deepcopy = True)
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        a,L = weak_form(sol_old,sol_test,sol,vort_old,flux_old,dt)    
   
        A = assemble(a)
        b = assemble(L)
        solve(A,k1.vector(),b)
        sol_old.assign(sol_n + 0.5*k1)      
        
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        a,L = weak_form(sol_old,sol_test,sol,vort_old,flux_old,dt)    
   
        A = assemble(a)        
        b = assemble(L)
        solve(A,k2.vector(),b)
        sol_old.assign(sol_n + 0.5*k2)
        
       
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        a,L = weak_form(sol_old,sol_test,sol,vort_old,flux_old,dt)    
   
        A = assemble(a)
        b = assemble(L)
        solve(A,k3.vector(),b)
        sol_old.assign(sol_n + k3)
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        a,L = weak_form(sol_old,sol_test,sol,vort_old,flux_old,dt)    
      
        A = assemble(a)      
        b = assemble(L)
        solve(A,k4.vector(),b)
        sol_old.assign(sol_n + 1/6*(k1 + 2*k2 + 2*k3 + k4))
        
        u_f,h_f = sol_old.split(deepcopy = True)
        q_f,flux_f  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        
        dif_q = errornorm(q_0,q_f)
        dif_u = errornorm(u_0,u_f)
        dif_h = errornorm(h_0,h_f)
        
        
        error_vec[it,:] = dif_q,dif_u,dif_h
        scalars[it,:] = diagnostics(q_f,u_f,flux_f,h_f)      
    
        if output:
            q_f.rename('q_f','q')
            qfile << q_f,t
            
        
    # Compute CFL condition 
    cfl = (max(u_0.vector()[:]))*dt/mesh.hmin()
    
    if cfl > 1.0:
        print('cfl condition is not satisfied')
        return
    
    if dif_u < 1e-1 and dif_h < 1e-1:
        # Move to next time step
        t += dt
        it = it + 1
    else:
            print('Solver diverges')
            return 

    return error_vec,scalars