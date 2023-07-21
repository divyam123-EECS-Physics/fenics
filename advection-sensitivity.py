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
    data_t = np.gradient(data, t, axis=t_axis)
    return data_x, data_t


def residual(u, x, t, nu):
    data_x, data_t = partials(u, x, t)
    pi = torch.pi

    eqn1 = (nu * data_x) + data_t 
    return np.abs(eqn1), data_x, data_t
    
def residual_jax(u, u_partials, nu):
    data_x, data_t  = u_partials
    pi = jnp.pi
    eqn1 = (nu * data_x) + data_t 
    res = jnp.abs(eqn1.astype('float64'))
    return jnp.mean(res, axis=(1, 2)), data_x, data_t

def sensitivity_analysis(u, x, t, gt_nu, min_noise=-1, max_noise=1, num_noise=20):
    u_partials = partials(u, x, t)
    noise = np.linspace(min_noise, max_noise, num_noise) * gt_nu
    gt_residual = residual_jax(u, u_partials, gt_nu)
    gt_nu = np.asarray(gt_nu)
    nu = np.broadcast_to(gt_nu, (noise.shape[0])) + noise
    nu_residuals = jax.vmap(residual_jax,
                        in_axes=(None, None,0))(u, u_partials, nu)

    return gt_residual, nu_residuals, noise

def plot_residuals(residuals, t, x, nu, savedir):
    residuals = np.mean(residuals, axis=(2, 3))
    fig = plt.figure()
    for seed in range(0,residuals.shape[0],3):
        plt.plot(t, residuals[seed, :], label = 'init_cond:'+str(seed))
    plt.yscale('log')
    plt.xlabel('Time')
    plt.legend()
    plt.ylabel('Residual')
    plt.title('nu = ' + str(nu) + f', tdim = {t.shape[0]}, resolution = {x.shape[0]}')
    plt.savefig(plotdir + '/burgers_residuals/'+ 'nu=' + str(nu) + f'_tdim={t.shape[0]}_resolution={x.shape[0]}.png')
    plt.clf()
def plot_sensitivity_analysis(gt_residuals, nu_residuals, noise, nu, savdir):
    plot_sensitivity_inner(gt_residuals, nu_residuals, noise, 'beta', savdir, nu)
    
def plot_sensitivity_inner(gt_residuals, noise_residuals, noise, param, save_dir, nu):
    fig = plt.figure()
    plt.plot(gt_residuals, label='Ground Truth', linestyle='-.')
    for i in range(len(noise_residuals)):
        plt.plot(noise_residuals[i], label=f'{noise[i]:.2e}')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.title(f'beta = {nu} - {param} noise')
    # plt.legend()
    plt.legend(loc="upper right", borderaxespad=0)
    plt.savefig(plotdir  + f'{param}_noise_beta=_{nu}.png')
    plt.close()


import pdb

plt.rcParams['figure.figsize'] = (10, 8)
random.seed(0)
np.random.seed(0)


# plotdir = '/home/divyam123/noether_work/noether-networks/scripts/plot_dir'#args.plotdir
# datadir = '/data/divyam123/fenics_fixed' 
# datadir = '/data/nithinc/pdebench/advection'
# datadir = '/data/divyam123/advection_fenics_fixed'
# plotdir = '/home/divyam123/noether_work/noether-networks/scripts/plot_dir/advection_higher_res_sensitivity/'#args.plotdir
# datadir = '/data/divyam123/advection_higher_res'
plotdir = '/home/divyam123/fenics/advection_sensitivity/'#args.plotdir
datadir = '/home/divyam123/fenics/advection'
parameter_combos = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and f.endswith('.hdf5')]
for f in parameter_combos:
    d = h.File(datadir + '/' + f)
    nu =  float(f[:-5].split('a')[-1])
    data = np.asarray(d['tensor'][0])
    x = np.asarray(d['x-coordinate'])
    t = np.asarray(d['t-coordinate'])#[1:]
    data = data[:,:,None]
    curr_outdir = plotdir#os.path.join(plotdir, f'nu={nu}')
    gt_residual, nu_residual, noise = sensitivity_analysis(
        data.astype('float64'), x.astype('float64'), t.astype('float64'), nu)
    # pdb.set_trace()
    plot_sensitivity_analysis(gt_residual[0], nu_residual[0], noise, nu, curr_outdir)




