
Solve 2D Shallow-water equation in unit square plane with periodic boundary conditions.

The initial condition is given by:


H = 10.0 + (1/1000.0)*sin(2*pi*x)*cos(y)

u = (1/1000.0)*sin(2*pi*x)*sin(y)

v = (1/1000.0)*2*pi*cos(2*pi*x)*cos(y)  


The Coriolis and gravity term are given respectively by:


f = g = 10.0 


The time step is adapted in order to mantain the CFL condition equal to 0.9.

The domain is discretized with a right-biased triangular mesh.

The Rossby number is Ro = U/Lf = 10^-4. Therefore the system is dominated by Coriolis forces and inertial forces can be neglected.









