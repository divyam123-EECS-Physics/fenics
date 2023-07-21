from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dolfin import *
import pdb

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return bool(((x[0] < DOLFIN_EPS) or (x[0] > 1 + DOLFIN_EPS)) and on_boundary)
        return bool((x[0] < DOLFIN_EPS) or (x[0] > (1.0 + DOLFIN_EPS)))# and on_boundary)
        # return bool(- DOLFIN_EPS < x[0] < DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1
x_coordinate = np.linspace(0,1,1024)
t_coordinate = np.linspace(0,2,8000)

for _nu_ in np.linspace(0.01,2,200,dtype = np.float128):
    solution = np.zeros((30,201,1024))#,dtype = np.float128)
    for i in range(30):
        # pdb.set_trace()
        Amp = [np.float128(np.random.uniform()), np.float128(np.random.uniform())]
        k = np.random.randint(8, size=2) #* 2 * np.pi
        phi = [np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)]

        initial_cond = lambda x: (Amp[0] * np.sin((2*np.pi*k[0]* x) + (phi[0]))) + (Amp[1] * np.sin((2*np.pi*k[1]* x) + (phi[1])))

        nu = _nu_# / np.pi
        timestep = t_coordinate[1]

        t = t_coordinate[1]
        end = 2.0
        j = 1
        solution[i,0,:] =  initial_cond(x_coordinate)
        while (t <= end):
            t += timestep
            solution[i,j,:] = np.float128(initial_cond(x_coordinate - (nu * t)))
            j += 1
    tensor = solution
    f = h5py.File('/data/divyam123/advection_analytical/1D_Advection_Sols_beta{}.hdf5'.format(_nu_), 'w')
    f['x-coordinate'] = x_coordinate
    f['t-coordinate'] = t_coordinate
    f['tensor'] = tensor
    f.close()
