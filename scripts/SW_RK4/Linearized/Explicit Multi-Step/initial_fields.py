#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:39:20 2020

@author: simo94
"""




from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def initial_fields(mesh,space_u,space_h,deg_u,deg_h,case):
    
    CG_u = FunctionSpace(mesh,space_u,deg_u+3)
    CG_h = FunctionSpace(mesh,space_h,deg_h+3)
    u_0 = Function(CG_u)
    h_0 = Function(CG_h)
    
    if case == '1D':
        
        expr_u = Expression(('sin(2.0*pi*x[1])/1000','0.000000'),element = CG_u.ufl_element())
        expr_h = Expression('10.0 + 1.0/(2.0*pi*1000)*cos(2.0*pi*x[1])',element = CG_h.ufl_element())
        
    else:
    
        expr_u = Expression(('sin(pi*x[1])','0.0000000'), element = CG_u.ufl_element())
        expr_h = Expression('10.000 + sin(2*pi*x[0])*sin(2*pi*x[1])',element = CG_h.ufl_element())
    
    
    u_0 = interpolate(expr_u,CG_u)
    h_0 = interpolate(expr_h,CG_h)
    
    return u_0,h_0