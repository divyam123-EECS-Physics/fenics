import random
import h5py as h
import numpy as np
import os
import argparse
import jax.numpy as jnp
import jax
import torch
# from .scripts.burgers_common import Burgers1D
import matplotlib.pyplot as plt
random.seed(0)
def partials(data, x, t):
    x_axis = -2
    t_axis = -3
    data_x = np.gradient(data, x, axis=x_axis)
    data_x_usqr = np.gradient(data *  data / 2, x, axis=x_axis)
    data_xx = np.gradient(data_x, x, axis=x_axis)
    data_t = np.gradient(data, t, axis=t_axis)
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
    res = jnp.abs(eqn1.astype('float64'))
    return jnp.mean(res, axis=(1, 2))

def sensitivity_analysis(u, x, t, gt_nu, min_noise=-1, max_noise=1, num_noise=20):
    u_partials = partials(u, x, t)
    noise = np.linspace(min_noise, max_noise, num_noise) * gt_nu
    gt_residual = residual_jax(u, u_partials, gt_nu)
    gt_nu = np.asarray(gt_nu)
    nu = np.broadcast_to(gt_nu, (noise.shape[0])) + noise
    nu_residuals = jax.vmap(residual_jax,
                        in_axes=(None, None,0))(u, u_partials, nu)

    return gt_residual, nu_residuals, noise

def plot_sensitivity_analysis(gt_residuals, nu_residuals, noise, nu, savdir):
    # gt_residuals: n_timesteps, n_spatial
    # x_residuals: n_noise, n_timesteps, n_spatial
    # noise: n_noise

    # First plot du noise
    plot_sensitivity_inner(gt_residuals, nu_residuals, noise, 'nu', savdir, nu)
    
def plot_sensitivity_inner(gt_residuals, noise_residuals, noise, param, save_dir, nu):
    fig = plt.figure()
    plt.plot(gt_residuals, label='Ground Truth', linestyle='-.')
    for i in range(len(noise_residuals)):
        plt.plot(noise_residuals[i], label=f'{noise[i]:.2e}')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.title(f'nu = {nu} - {param} noise')
    plt.legend(loc="upper right", borderaxespad=0)
    plt.savefig(save_dir + '/' + f'{param}_noise_nu=_{nu}.png')
    plt.close()


# import pdb

plt.rcParams['figure.figsize'] = (10, 8)
random.seed(0)
np.random.seed(0)


plot_dir = '/home/divyam123/fenics/'

sensitivity_path_org = 'pde_bench_original_download_sensitivity'
datadir_org = '/data/divyam123/pdebench_download'

sensitivity_path_nithin = 'pde_bench_nithin_sensitivity'
datadir_nithin = '/data/nithinc/pdebench/burgers'

sensitivity_path_fenics = 'pde_bench_fenics_sensitivity'
datadir_fenics = '/data/divyam123/fnics_two_sines'

# files = [
#     (sensitivity_path_org, datadir_org,1),
#     (sensitivity_path_nithin, datadir_nithin,1),
#     (sensitivity_path_fenics, datadir_fenics,0),
# ]

# files = [
#     ('fenics_1.5max_sensitivity', '/data/divyam123/fenics_1.5max',0),
# ]

files = [
    ('burgers_sensitivity','/home/divyam123/fenics/burgers',0)
]

for sensitivity_path, datadir, choice in files: 
    parameter_combos = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and f.endswith('.hdf5')]
    for f in parameter_combos:
        d = h.File(datadir + '/' + f)
        nu =  float(f[:-5].split('u')[-1])
        data = np.asarray(d['tensor'][0])
        x = np.asarray(d['x-coordinate'])
        t = np.asarray(d['t-coordinate'])[1:] if choice == 1 else np.asarray(d['t-coordinate'])
        data = data[:,:,None]
        curr_outdir = plot_dir + ''#os.path.join(plotdir, f'nu={nu}')
        gt_residual, nu_residual, noise = sensitivity_analysis(
            data, x, t, nu)
        # pdb.set_trace()
        save_path = plot_dir + sensitivity_path
        plot_sensitivity_analysis(gt_residual, nu_residual, noise, nu, save_path)




