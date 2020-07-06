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


## User expression for merging vortex problem 

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
            
            value[0] =  -(1.0/100.0)*dy*(exp(-(dxa*dxa + dy*dy)/(2.0*self.sigma)) + exp(-(dxb*dxb + dy*dy)/(2.0*self.sigma)))/(2.0*pi*self.sigma**2)
            value[1] = (1.0/100.0)*(dxa*exp(-(dxa*dxa + dy*dy)/(2.0*self.sigma)) + dxb*exp(-(dxb*dxb + dy*dy)/(2.0*self.sigma)))/(2.0*pi*self.sigma**2)
            value[2] = 10.0  + (1.0/100.0)*(1.0/(2.0*pi*self.sigma))*(exp(-(dxa*dxa + dy*dy)/(2.0*self.sigma)) +  exp(-(dxb*dxb + dy*dy)/(2.0*self.sigma)))

            
    def value_shape(self):
        return (3,)


form_compiler_parameters = {"quadrature_degree": 6}


def diagnostics(vort,u,flux,h,S,dt,t):

    g = Constant(10.0)  # gravity
    #vort_mid = (vort0 + vort)/2
    #u_mid = (u0 + u)/2
    #h_mid = (h0 + h)/2
    
    
    S1,S2,S3 = split(S)
    # Energy
    E = assemble(0.5*(h*inner(u,u) + g*h**2)*dx)
    #E = assemble(0.5*(h*inner(u,u) + g*h**2)*dx) - assemble(0.5*(h0*inner(u0,u0) + g*h0**2)*dx)- dt*(assemble(h*(u_mid[0]*S1 + u_mid[1]*S2)*dx) + assemble(S3*(0.5*inner(u_mid,u_mid) + g*h_mid)*dx))
          
    # Enstrophy    
    Ens = assemble((vort**2)*h*dx)
    #Ens = assemble((vort**2)*h*dx) - assemble((vort0**2)*h0*dx)- dt*(assemble(2*vort_mid*(-S1.dx(1) + S2.dx(0))*dx) - assemble((vort_mid**2)*S3*dx)) 
    
    # Absolute vorticity 
    Absv = assemble(vort*h*dx)
        
    # Mass
    M = assemble(h*dx) #- t*assemble(S3*dx)

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
    
   f = Constant(50.0*2*pi)
    
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
   
   
   # Source term 
   S_momentum = dt*(u_test[0]*S1 + u_test[1]*S2)*dx
   
   
   # Anticipated potential vorticity
   # This removes inertia-gravity waves so that we can apply our moving mesh
   # strategy
   
   #tau = dt/2.0
   # 1st component of Q
   #Q_momentum =  + tau*(u_old[0]*(vort_old.dx(0)*flux_old[0] + vort_old*flux_old[0].dx(0)) \
   #                    + u_old[1]*(vort_old.dx(1)*flux_old[0] + vort_old*flux_old[0].dx(1)))*u_test[1]*dx \
   #             - tau*(u_old[0]*(vort_old.dx(0)*flux_old[1] + vort_old*flux_old[1].dx(0)) \
   #                    + u_old[1]*(vort_old.dx(1)*flux_old[1] + vort_old*flux_old[1].dx(1)))*u_test[0]*dx
   
   
   L += C_momentum + A_momentum + S_momentum #+ Q_momentum  
    
   #### Continuity equation #### 

    
   rhs_continuity = -dt*(h_test*div(flux_old))*dx + dt*h_test*S3*dx
   L += rhs_continuity

    
   return L


def solver(mesh,W1,W2,u_0,h_0,dt,tf,output,lump,case):
    
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
    
    
    Z = FiniteElement('CG',mesh.ufl_cell(),4)
    W_elem = MixedElement([Z,Z,Z])
    Z = FunctionSpace(mesh,W_elem)
    
    ## Source term 
    S = Function(Z)   
           
    
    ## 1D case
    
    if case == '1D':
        expr = Expression(('sin(4.0*pi*x[1])/1000','0.0',
                       '10.0 + 1.0/(4.0*pi*1000)*cos(4.0*pi*x[1])'),element = W1.ufl_element())
        
        print('1D')
        # expr_u = Expression(('sin(4.0*pi*x[1])','0.0'),element = CG_u.ufl_element())
        # expr_h = Expression('10.0 + 1.0/(4.0*pi)*cos(4.0*pi*x[1])',element = CG_h.ufl_element())
        
        
   
        # Source term for 1D case 
        S_exp = Expression(('0.0','0.0','0.0'),degree=4)  
    
    else:
        
        expr = Expression(('sin(pi*x[1])','0.0000000','10.000 + sin(2*pi*x[0])*sin(2*pi*x[1])'),element = W1.ufl_element())
    
     
        # expr_u = Expression(('sin(pi*x[1])','0.00000'),element = CG_u.ufl_element())
        # expr_h = Expression('10.000 + sin(2*pi*x[0])*sin(pi*x[1])',element = CG_h.ufl_element())
        
         # Source term for 2D case on the right side
        S_exp = Expression(('10.0*2*pi*cos(2*pi*x[0])*sin(2*pi*x[1])',
                        '10.0*sin(pi*x[1]) + 10.0*sin(2*pi*x[0])*2*pi*cos(2*pi*x[1])',
                        '2*pi*cos(2*pi*x[0])*sin(pi*x[1])*sin(2*pi*x[1])'),degree=4)
   
    
    
    #expr = MyExpression(sigma = 0.01,xa = 0.4 ,xb = 0.6,yc = 0.5,element = W1.ufl_element())
    sol_old.interpolate(expr)
    
    #S_exp = Expression(('0.0','0.0','0.0'),degree=4)  
    S.interpolate(S_exp)

    
    t = 0.0
    nt = int(tf/dt) 
    it = 0 
    
    'Implementation of the Runge-Kutta method'
    
    sol_n = Function(W1)
    
    k1 = Function(W1)     
    k2 = Function(W1)
    k3 = Function(W1) 
    k4 = Function(W1)
   
    
    scalars = np.zeros(4*(nt+1)).reshape(nt+1,4)
    devs_vec = np.zeros(2*(nt+1)).reshape(nt+1,2) 
    
    #u_f,h_f = sol_old.split(deepcopy = True)
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
        u_0.rename('u_0','u')
        h_0.rename('h_0','h')
        ufile << u_0,t
        hfile << h_0,t
          
        
    while(it <= nt):    

        sol_n = sol_old.copy(deepcopy = True)
        
        q_old,flux_old  = diagnostic_vars(sol_old,sol_test,diag_test,diag_trial,diag_old)
        diag_n = diag_old.copy(deepcopy = True) 
        
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
        
        
        
        dif_u = errornorm(u_0,u_f,norm_type='l2', degree_rise=3)
        dif_h = errornorm(h_0,h_f,norm_type='l2', degree_rise=3)      
        
        devs_vec[it,:] = dif_u,dif_h
        
        t += dt
        scalars[it,:] = diagnostics(q_f,u_f,flux_f,h_f,S,dt,t)      
          
        # Compute CFL condition by looking at max(u_vector/h) for each cell
        # of the mesh 
        max_u = max(u_f.vector()[:])
        cfl = (max_u)*dt/mesh.hmin()
        
        if cfl > 1.0:
            print('cfl condition is not satisfied')
            
        
        # Move to next time step
        if dif_u > 1.0 or dif_h > 1.0: 
            
            Warning('The RK scheme diverges')
            print('Solver diverges')
            return 
        
        it = it + 1
        
        if output:
            u_f.rename('u_f','u')
            h_f.rename('h_f','h')
            q_f.rename('q_f','q')
            ufile << u_f,t
            hfile << h_f,t  
            qfile << q_f,t


    return devs_vec,scalars