# Solution of nonlinear 2D shallow Water equations with the Mixed Finite Element method

Steady State Solution of 2D nonlinerar RSWs using Mixed Finite Element (MFE) method and Runge-Kutta 4 (RK4) scheme.

We aim to solve the system of equations for the velocity field U = (u,v) and the height field D on a unit square domain with periodic boundary conditions.

u_t +  u*u_x + v*u_y - f*v + g*D_x = 0


v_t +  u*v_x + v*v_y + f*u + g*D_y = 0


D_t +  div(U*D) = 0

The nonlinear system is rewritten in vector invariant solve in order to ensure conservation of absolute vorticity and enstrophy. In the f-plane we can obtain local conservation of potential vorticity q.


We first discretize the equations in space using the Mixed Finite Element scheme, which represents the 3 unknowns (q,U,D) in compatible FEM spaces. The integration in time is then performed with a Runge-Kutta 4 scheme. 

The error introduced by the spatial discretization can be assessed by simulating a flow in geostrophic balance. The initial conditions is given by:

u0 = sin(4 pi y)  v0 = 0    
h0 = 10 + 1/4pi cos(4 pi y)

Based on the choice of the Finite Element Spaces, the discrete solution can be polluted by spurious oscillations of different nature. The balance of velocity and pressure degrees of freedom (DOFs) dof(U) = 2*dim(D) is a a necessary condition for the absence of spurious modes (Cotter and Shipton, 2012).

It is known that RT spaces on triangles have a surplus of pressure degrees of freedom (DOFs) and consequently have
spurious inertia–gravity modes. Infact the velocity space RT1 has 1.5 DOFs, while the height field space DG0 has dimension 1.0. On the other hand, BDM spaces for velocity have a deﬁcit of pressure DOFs and consequently the solution is polluted by spurious Rossby modes.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Refinement strategy:


Refinement of the Mesh will be realized according to h and r-adaptive strategy.

h-adaptive:

In the file refinement.py the input mesh is refined iteratively according to the input 'n_iter'. It is also possible to specify the ratio of cell to be defined by 'ref_ratio'.

The refinement strategy is based on the evaluation of a monitor function across all the cells of the domain. For a-priori refinement we exploit the known analytic expression of the exact steady-state solution and increase the resolution of the mesh in the regions with highest gradient. A posteriori refinement relies on an error estimate using a reconstructed value for u and h. This reconstruction method has been proved to work for hyperbolic systems of conservation law in 1D for Lagrangian Finite elements. An extension to 2D domain with MFEM Spatial discretization is required.


r-adaptive:

r-adaptive schemes dislocates a fixed number of mesh nodes over time upon equidistributing a monitor function. 
In the problem we are examining, we exploit the invariance of the solution along the x-axis, so that we can compute a 1D equidistributing mesh along the y-axis. The equidistribution strategy is implemented using De Boor's algorithm.


If a-priori knowledge of the exact solution is used in the monitor function, the equidistributing mesh is only computed at the initial iteration. If the residual is used to evaluate the monitor function, then the mesh coordinates change at every time step and interpolation of the old solution onto the new mesh is required. This step might rise issues if we want to ensure conservation of Mass, Absolute Vorticity, Energy, Enstrophy.


Add new line
