# SW_RK4

Steady State Solution of 2D nonlinerar RSWs using Mixed Finite Element (MFE) method and Runge-Kutta 4 (RK4) scheme. 

The initial condition for velocity and height field is given by:

u0 = (sin(4*pi*y),0)    
h0 = 10 + 1/4pi*cos(4*pi*y)

The domain is a unit square with periodic boundary conditions. 

Refinement of the Mesh will be realized according to h and r-adaptive strategy.

h-adaptive: a priori and posteriori refinement 
r-adaptive: 1+1D method with optimally-interpolant-L2 monitor function and optimal transport moving mesh method 

