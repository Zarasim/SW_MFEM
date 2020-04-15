
Solve 2D Shallow-water equation in unit square plane with periodic boundary conditions.

The initial condition is given by:


H = 10.0 + sin(2 pi x)*sin(pi y)

u = sin(pi*y)

v = 0.0  

In order to have this as a steady-state solution for our system, we have to add source terms to the discretized momentum and continuity equations. This way Mass, Energy and Enstrophy are no longer conserved. 


The Coriolis and gravity term are f = g = 10.0. 


The time step is set to 0.0005. The domain is discretized with a right-biased triangular mesh.








