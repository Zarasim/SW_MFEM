## README plot

uniform folder contain plots for uniform refinement
mov_mesh folder contain plots for 1+1D moving mesh method with monitor function based on exact gradient 


Deviations from initial conditions are dampend oscillatory. Is this due to computational modes for RT1 - DG0 Function spaces ?

Relative deviations of physical invariants are also oscillatory and of order 10^-7. The energy though is dissipated in the small scale.

Convergence rate does not improve significantly. This might be due to the lack of FE residual upper bound for r-adaptive schemes. 
