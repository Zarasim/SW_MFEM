# SW_RK4

Steady State Solution of 2D nonlinerar RSWs using Mixed Finite Element (MFE) method and Runge-Kutta 4 (RK4) scheme. 

The initial condition for velocity and height field is given by:

u0 = (sin(4 pi y),0)    
h0 = 10 + 1/4pi cos(4 pi y)

The domain is a unit square with periodic boundary conditions. 

Refinement of the Mesh will be realized according to h and r-adaptive strategy.

h-adaptive:

In the file refinement.py the input mesh is refined iteratively according to the input 'n_iter'. It is also possible to specify the ratio of cell to be defined by 'ref_ratio'.

The refinement strategy is based on the evaluation of a monitor function across all the cells of the domain. For a-priori refinement we exploit the known analytic expression of the exact steady-state solution and increase the resolution of the mesh in the regions with highest gradient. A posteriori refinement relies on an error estimate using a reconstructed value for u and h. This reconstruction method has been proved to work for hyperbolic systems of conservation law in 1D for Lagrangian Finite elements. An extension to 2D domain with MFEM Spatial discretization is required.


r-adaptive:

r-adaptive schemes dislocates a fixed number of mesh nodes over time upon equidistributing a monitor function. 
In the problem we are examining, we exploit the invariance of the solution along the x-axis, so that we can compute a 1D equidistributing mesh along the y-axis. The equidistribution strategy is implemented using De Boor's algorithm.


If a-priori knowledge of the exact solution is used in the monitor function, the equidistributing mesh is only computed at the initial iteration. If the residual is used to evaluate the monitor function, then the mesh coordinates change at every time step and interpolation of the old solution onto the new mesh is required. This step might rise issues if we want to ensure conservation of Mass, Absolute Vorticity, Energy, Enstrophy.


