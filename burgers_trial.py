from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import h5py



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
x_coordinate = np.linspace(0,1,2048)
t_coordinate = np.linspace(0,2,201)

for _nu_ in np.linspace(0.01,10,200):
    solution = np.zeros((15,201,2048))
    for i in range(15):
        Amp = [np.random.uniform(), np.random.uniform()]
        k = np.random.randint(8, size=2) * 2 * np.pi
        phi = [np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)]
        # Create periodic boundary condition
        pbc = PeriodicBoundary()

        n = 2048
        mesh = UnitIntervalMesh(n)
        V = VectorFunctionSpace(mesh, "P", 1, constrained_domain = pbc) # setting up the space/mesh
        
        initial_cond = "{}*sin({}*x[0] + {}) + {}*sin({}*x[0] + {})".format(Amp[0], k[0], phi[0],Amp[1], k[1], phi[1])  ## defnining initial cond
        u = project(Expression((initial_cond,), degree = 1),  V)

        u_next = Function(V) 
        v = TestFunction(V)

        nu = Constant(_nu_) / np.pi # constant definintion 

        timestep = Constant(0.01)

        F = (inner((u_next - u)/timestep, v) + inner(grad(u_next)*u_next, v)+ nu*inner(grad(u_next), grad(v)))*dx
        ### ^^^^^ definind non linear variational problem

        # u0 = Constant(0.0)
        # dbc = DirichletBoundary()
        # bc = DirichletBC(V, [u0], dbc)

        t = 0.01
        end = 2.01
        j = 1
        solution[i,0,:] =  u.vector()[:]
        while (t < end):

            solve(F == 0, u_next, [], 
                solver_parameters={"newton_solver":{"relative_tolerance":1e-3,
                                                    "absolute_tolerance":1e-6}}) ## solving non linear problem

            u.assign(u_next)
            t += float(timestep)
            solution[i,j,:] = u.vector()[:]
            j += 1

    f = h5py.File('/data/divyam123/burgers_higher_res/1D_Burgers_Sols_Nu{}.hdf5'.format(_nu_), 'w')
    f['x-coordinate'] = x_coordinate
    f['t-coordinate'] = t_coordinate
    f['tensor'] = solution
    f.close()
