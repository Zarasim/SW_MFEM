t#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:39:03 2020

Apply local mesh refinement using gradient monitor function 

@author: simo94
"""

from fenics import *
from mshr import *
import numpy as np

def refinement(mesh,n_ref,ref_ratio):
    
    # Compute gradient of solution 
    source_str = 'sqrt((-sin(4*pi*x[1]))**2 + (4*pi*cos(4*pi*x[1]))**2)'
    
    # Define lambda function x with source_str using eval() built-in function
    source = eval('lambda x: ' + source_str)
    
    
    for level in range(n_ref):
            
        h = np.array([c.h() for c in cells(mesh)])  # diameter per each cell
        K = np.array([c.volume() for c in cells(mesh)])  # Volume per each cell     
    
        # source term evaluated at midpoint of each cell
        R = np.array([abs(source([c.midpoint().x(),c.midpoint().y()])) for c in cells(mesh)])  
        gamma = np.sqrt(K)*R

        # Compute error estimate
        E = sqrt(sum([g*g for g in gamma]))

        print('Level %d: E = %g ' %(level,E))
        
        THETA = E/sqrt(mesh.num_cells())


        # Mark cells for refinement
        cell_markers = MeshFunction('bool',mesh,mesh.topology().dim())   

            
        if ref_ratio:
            gamma_0 = sorted(gamma,reverse = True)[int(len(gamma)*ref_ratio)]
            gamma_0 = MPI.max(mesh.mpi_comm(),gamma_0)

            for c in cells(mesh):
                cell_markers[c] = gamma[c.index()] > gamma_0
                
        else:  # apply refinement according to an equidistribution condition  
            for c in cells(mesh):
                cell_markers[c] = gamma[c.index()] > THETA


        # Refine mesh 
        mesh = refine(mesh,cell_markers)
        
    return mesh 