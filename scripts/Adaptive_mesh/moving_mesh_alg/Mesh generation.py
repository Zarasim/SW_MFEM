#!/usr/bin/env python
# coding: utf-8

# # Mesh in Fenics 
# 
# There exist several options for creating a mesh on Fenics. First, mshr is the official mesh generator component that generate simplicial geometries described by Constructive Solid Geometry (CSG) and generate the mesh using CGAL and Tetgen as backend.
# 
# A second alternative consist of creating the mesh using an external sofware as Gmsh.It offers a free-integrated solution between CAD engine, mesh generator and post-processing. Its kernel is composed of 4 modules (geometry, mesh generator, solver and post-processing) and its own scripting language.
# 
# In order to be light and fast, Gmsh is completely written in C++. It uses BLAS and LAPACK for most of
# the basic linear algebra. The Kernel makes extensive use of the Standard Template Library for all
# data containers. The graphical interface has been built with FLTK and OpenGL, this reduces the
# memory footprint.
# 
# Gmsh can be built as a library, providing an API for accessing geometry, mesh and post-processing,
# or can be installed as a stand-alone software.

# ##  Mesh generation using dolfin 1.6

# In[1]:


#This is a script for generating nonuniform meshes
    
from dolfin import *
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


nr = 10 # 10 horizontal elements 
nt = 10 # 10 vertical elements 


# the data is store as dolfin.cpp.rectanglemesh
mesh = RectangleMesh(Point(0.0,0.0),Point(5.0,1.0),nr,nt)


# In[3]:


# mesh.coordinates() gives in sequence the node location along horizontal elements from Point(x_a,y_a)

r = mesh.coordinates()[:,0]
t = mesh.coordinates()[:,1]

s = 1.3


# get finer mesh towards x = x_a
# points in y-dir equidistant

def denser(x,y):  
    a = x[0]
    b = x[-1]
    return [a + (b-a)*((x-a)/(b-a))**s,y]


r_bar,t_bar = denser(r,t)
rt_bar_coord = numpy.array([r_bar,t_bar]).transpose()


mesh.coordinates()[:] = rt_bar_coord


# In[ ]:


plot(mesh)


# In[ ]:


# Define mesh mapping from rectangle to cylinder

Theta = pi/2
def cylinder(r,t):
    return [r*numpy.cos(Theta*t),r*numpy.sin(Theta*t)]


x_hat,y_hat = cylinder(r_bar,t_bar)
xy_hat_coord = numpy.array([x_hat,y_hat]).transpose()

mesh.coordinates()[:] = xy_hat_coord


# In[ ]:


plot(mesh)


# ## Generate ustructured Meshes with mshr

# In this case the grid is unstructured 

# In[8]:


from mshr import *


# In[ ]:


# Define 2D geometry
# - applies a difference operator 

domain = Rectangle(Point(0.,0.),Point(5.,5.)) - Circle(Point(1,4),.5) - Circle(Point(4,4),.5)

# Define boundary before generating mesh 

# Generate and plot mesh using cgal library as backend 
mesh2d = generate_mesh(domain,30)   

plot(mesh2d)


# In[9]:


# Define 2D geometry
# - applies a difference operator 

domain = Rectangle(Point(0.,0.),Point(5.,5.))

# Define boundary before generating mesh 

# Generate and plot mesh using cgal library as backend 
mesh2d = generate_mesh(domain,10)   

plot(mesh2d)


# ## Refinement of unstructured mesh

# In[ ]:


source_str = 'sqrt((10 + 1/(4*pi)*cos(4*pi*x[1]))**2 + (-sin(4*pi*x[1]))**2 + (4*pi*cos(4*pi*x[1]))**2)'
# Define lambda function x with source_str using eval() built-in function
source = eval('lambda x: ' + source_str)


# In[11]:


TOL = 1e-3
REFINE_RATIO = 0.30 # Refine 30% of cells in each iteration
MAX_ITER = 10 # max number of iterations


# In[13]:


for level in range(MAX_ITER):
    
    h = np.array([c.h() for c in cells(mesh2d)])  # diameter per each cell
    K = np.array([c.volume() for c in cells(mesh2d)])  # Volume per each cell     
    
    # source term evaluated at midpoint of each cell
    R = np.array([abs(source([c.midpoint().x(),c.midpoint().y()])) for c in cells(mesh2d)])  
    gamma = h*np.sqrt(K)*R

    # Compute error estimate
    E = sum([g*g for g in gamma])

    # mpi computes the sum using parallel computation
    E = sqrt(MPI.sum(mesh2d.mpi_comm(),E))
    print('Level %d: E = %g (TOL = %g)' %(level,E,TOL))

    # check convergence
    if E < TOL:
        info('success')
        break

    # Mark cells for refinement
    cell_markers = MeshFunction('bool',mesh2d,mesh2d.topology().dim())

    gamma_0 = sorted(gamma,reverse = True)[int(len(gamma)*REFINE_RATIO)]
    gamma_0 = MPI.max(mesh2d.mpi_comm(),gamma_0)

    for c in cells(mesh2d):
        cell_markers[c] = gamma[c.index()] > gamma_0

    # Refine mesh 
    mesh2d = refine(mesh2d,cell_markers)


# In[ ]:


File("my_mesh.xml") << mesh2d


# ## Import mesh from GMSH

# In case of complicated geometries, a mesh can be imported from an external GMSH software .geo ->.msh, which must be first converted into xml using dolfin-convert. Alternatively, install the software meshio for the import.

# In[1]:


import os
os.system('dolfin-convert mesh_1.msh mesh_1.xml')

# dolfin.cpp.mesh.Mesh can be handled as the rectangle generated above
mesh = Mesh('my_mesh.xml')

mesh

# subdomains and boundaries make no sense

subdomains = MeshFunction("size_t", mesh, "mesh_1_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "mesh_1_facet_region.xml")


# In[ ]:


V = FunctionSpace(mesh,'Lagrange',1)


# To access the boudnary layers you can use the string assigned to each boundary (e.g. 'inlet','outlet' etc...)

# In[ ]:


bcs = [DirichletBC(V, 5.0, boundaries, ),# of course with your boundary
    DirichletBC(V, 0.0, boundaries, 0)]

