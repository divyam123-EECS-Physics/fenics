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

for _nu_ in np.linspace(0.01,2,200):
    solution = np.zeros((5,8000,1024))
    for i in range(5):
        # pdb.set_trace()
        Amp = [np.random.uniform(), np.random.uniform()]
        k = np.random.randint(8, size=2) * 2 * np.pi
        phi = [np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)]
        # Create periodic boundary condition
        pbc = PeriodicBoundary()
        n = 1023
        mesh = UnitIntervalMesh(n)
        V = FunctionSpace(mesh, "P", 1, constrained_domain=pbc)
        initial_cond = "{}*sin({}*x[0] + {}) + {}*sin({}*x[0] + {})".format(Amp[0], k[0], phi[0],Amp[1], k[1], phi[1])

        print("i: ", i, "initial_condition: ", initial_cond)

        u0 = Constant(1)
        # bc = []
        u = interpolate(Expression((initial_cond,)[0], degree = 1),  V)


        u_next = TrialFunction(V)
        v = TestFunction(V)

        # dbc = DirichletBoundary()
        # bc0 = DirichletBC(V, u, pbc)
        # # bc = [bc0]
        # bc = []

        nu = Constant(_nu_)# / np.pi
        timestep = Constant(t_coordinate[1])
        F = ((((u_next - u ) * v) / timestep )) * dx
        F += ((nu * grad(u_next) * v))[0]* dx
        a, L = lhs(F), rhs(F)

        u_next = Function(V)

        t = t_coordinate[1]
        end = 2.0
        j = 1
        solution[i,0,:] =  u.compute_vertex_values()
        while (t <= end):
            print(j)
            # solve(F == 0, u_next, bc)

            solve(a == L, u_next, [], form_compiler_parameters={"relative_tolerance":1e-3,
                                                        "absolute_tolerance":1e-6})

            u.assign(u_next)
            t += float(timestep)
            solution[i,j,:] = u.compute_vertex_values()
            j += 1
    tensor = solution
    f = h5py.File('/data/divyam123/advection_cfl_res_1024/1D_Advection_Sols_beta{}.hdf5'.format(_nu_), 'w')
    f['x-coordinate'] = x_coordinate
    f['t-coordinate'] = t_coordinate
    f['tensor'] = tensor
    f.close()
