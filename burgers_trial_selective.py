from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pdb
import jax.numpy as jnp
import jax

def partials(data, x, t):
    data_temp = data.reshape((data.shape[0], data.shape[1], 1))
    x_axis = -2
    t_axis = -3
    data_x = np.gradient(data_temp, x, axis=x_axis)
    data_x_usqr = np.gradient(data_temp *  data_temp / 2, x, axis=x_axis)
    data_xx = np.gradient(data_temp, x, axis=x_axis)
    data_t = np.gradient(data_temp, t, axis=t_axis)
    return data_x, data_x_usqr, data_xx, data_t

def residual(u, x, t, nu):
    data_x, data_x_usqr, data_xx, data_t = partials(u, x, t)
    pi = np.pi
    eqn1 = data_x_usqr + data_t - ((nu/pi) * data_xx)
    return np.abs(eqn1)

def residual_jax(u, u_partials, nu):
    data_x, data_x_usqr, data_xx, data_t  = u_partials
    pi = jnp.pi
    eqn1 = data_x_usqr + data_t - ((nu/pi) * data_xx)
    res = jnp.abs(eqn1)
    return jnp.mean(res, axis=(1, 2))

def sensitivity_analysis(u, x, t, gt_nu, min_noise=-1, max_noise=1, num_noise=20):
    pdb.set_trace()
    u_partials = partials(u, x, t)
    noise = np.linspace(min_noise, max_noise, num_noise) * gt_nu
    gt_residual = residual(u, u_partials, gt_nu)
    gt_nu = np.asarray(gt_nu)
    nu = np.broadcast_to(gt_nu, (noise.shape[0])) + noise
    nu_residuals = jax.vmap(residual_jax,
                        in_axes=(None, None,0))(u, u_partials, nu)

    return gt_residual, nu_residuals, noise

def minimizer(data, x, t, nu):
    gt_residual, nu_residual, noise = sensitivity_analysis(data, x, t, nu)
    ground_truth = gt_residual[175]
    noisy_truths = []
    for i in range(len(noise_residuals)):
        noisy_truths.append(noise_residuals[i][175])
    if np.min(noisy_truths) == ground_truth:
        return False if noise[np.argmin(noisy_truths)] != nu else True
    elif np.min(noisy_truths) < ground_truth:
        return False
    elif np.min(noisy_truths) > ground_truth:
        return True
        

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)
        return bool(x[0] < 1 and x[0] > 0 and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x,y):
        y[0] = x[0] - 1.0

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[0] < DOLFIN_EPS or x[0] > (1.0 - DOLFIN_EPS)) and on_boundary)

x_coordinate = np.linspace(0,1, 1024)
t_coordinate = np.linspace(0,2, 1024)

for _nu_ in np.logspace(0.0001,1,200):
    solution = np.zeros((30,1024,1024))
    for i in range(30):
        Amp = [np.random.uniform(), np.random.uniform()]
        k = np.random.randint(8, size=2) * 2 * np.pi
        phi = [np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)]
        # Create periodic boundary condition
        pbc = PeriodicBoundary()

        n = 1023
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

        u0 = Constant(0.0)
        dbc = DirichletBoundary()
        bc = DirichletBC(V, [u0], dbc)

        t = 0.01
        end = 2.01
        j = 1
        solution[i,0,:] =  u.vector()[:]
        while (t < end):

            solve(F == 0, u_next, bc, 
                solver_parameters={"newton_solver":{"relative_tolerance":1e-3,
                                                    "absolute_tolerance":1e-6}}) ## solving non linear problem

            u.assign(u_next)
            t += float(timestep)
            solution[i,j,:] = u.vector()[:]
            j += 1

    f = h5py.File('/data/divyam123/fenics_1.5max/1D_Burgers_Sols_Nu{}.hdf5'.format(_nu_), 'w')
    f['x-coordinate'] = x_coordinate
    f['t-coordinate'] = t_coordinate
    f['tensor'] = solution
    f.close()


