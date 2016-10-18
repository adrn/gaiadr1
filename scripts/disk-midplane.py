"""
    Use the posterior samples to measure the midplane of the disk.

    Requires: astropy, emcee, h5py, numpy, schwimmbad
    (all pip-installable packages)
"""

__author__ = "adrn <adrn@princeton.edu>"

# Standard library
import os

# Third-party
from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
from astropy import log as logger
import emcee
import h5py
import numpy as np
from scipy.misc import logsumexp
from schwimmbad import choose_pool

def ln_gaussian(x, mu, ivar):
    return -0.5 * (ivar * (x - mu)**2 - np.log(2*np.pi*ivar))

def ln_sample_weights(zs):
    return -2*np.log(np.abs(zs))

def ln_likelihood(z, z0, mixture_amps, mixture_ivars):
    """
    n : index of star
    j : index of posterior distance sample under interim prior
    k : index of model mixture component

    Parameters
    ----------
    z : array_like (n, j)
        z [pc] computed from distances samples using the interim prior.
    z0 : numeric
        Midplane of the disk
    mixture_amps : array_like (k-1,)
        Amplitudes of Gaussian mixture model components.
    mixture_ivars : array_like (k,)
        Inverse-variances of Gaussian mixture model components.

    """
    z = np.array(z)
    if z.ndim == 1:
        z = z[None]

    mixture_amps = np.array(mixture_amps)
    mixture_amps = np.concatenate((mixture_amps, [1-mixture_amps.sum()]))
    mixture_ivars = np.array(mixture_ivars)
    J = z.shape[1]

    gaussian_shit = ln_gaussian(z[...,None], z0, mixture_ivars[None,None,:]) # (n,j,k)
    ln_numer = logsumexp(gaussian_shit, b=mixture_amps[None,None,:], axis=2) # over k
    lf = logsumexp(ln_numer - ln_sample_weights(z), b=1/J, axis=1) # over j

    return lf

def ln_prior(z0, log_mixture_amps, log_mixture_ivars):
    lp = 0.

    if np.any(log_mixture_ivars[1:] > log_mixture_ivars[:-1]):
        return -np.inf

    if np.sum(np.exp(log_mixture_amps)) > 1:
        return -np.inf

    lp += ln_gaussian(z0, 0., 1/25**2) # variance of the midplane center is (25 pc)^2 with 0 mean

    return lp

def ln_posterior(p, z, K):
    z0 = p[0]
    log_mixture_amps = p[1:K] # yes, K because there are K-1 amps
    log_mixture_ivars = p[K:]

    lp = ln_prior(z0, log_mixture_amps, log_mixture_ivars)
    if np.isinf(lp):
        return -np.inf

    ll = ln_likelihood(z, z0, np.exp(log_mixture_amps), np.exp(log_mixture_ivars))

    return lp + ll.sum()

def main(pool, stacked_tgas_path, distance_samples_path,
         output_path=None, seed=42, overwrite=False, continue_sampling=False):

    if os.path.exists(output_path) and overwrite:
        os.remove(output_path)

    if continue_sampling and not os.path.exists(output_path):
        raise IOError("Can't continue if initial file doesn't exist!")

    # read TGAS FITS file
    tgas = fits.getdata(stacked_tgas_path, 1)
    n_tgas = len(tgas)
    logger.debug("Loaded {} stars from stacked TGAS file.".format(n_tgas))

    # read all of the distance samples
    with h5py.File(distance_samples_path) as f:
        dists = f['distance'][:,:128] # OMG

    # only take nearby stars
    idx = np.median(dists, axis=1) < 200.
    dists = dists[idx]
    tgas = tgas[idx]

    # HACK:
    dists = dists[::16]
    tgas = tgas[::16]
    logger.debug("{} stars after dist cut.".format(len(dists)))

    g = coord.Galactic(l=tgas['l']*u.degree, b=tgas['b']*u.degree, distance=1*u.pc)
    gc = g.transform_to(coord.Galactocentric(galcen_distance=8.*u.kpc, z_sun=0.*u.pc))
    z = gc.cartesian.z.value[:,None] * dists

    if not continue_sampling:
        # TODO: should really let user set this
        K = 3
        log_ivars = -2*np.log([120, 250., 1000.])
        log_amps = np.log([0.05, 0.9])

        # just test that these work
        p0 = np.concatenate(([-20.], log_amps, log_ivars))
        n_dim = p0.shape[0]

        n_walkers = len(p0) * 16
        logger.debug("{} MCMC walkers, {} params".format(n_walkers, len(p0)))
        p0s = emcee.utils.sample_ball(p0, p0 * 1e-3, size=n_walkers)
        for _p0 in p0s:
            assert np.isfinite(ln_posterior(p0, z[:128], K))

    else:
        with h5py.File(output_path, 'r') as f:
            pre_chain = f['chain'][:]
            p0s = pre_chain[:,-1]
            K = f['chain'].attrs['K']

        n_walkers, n_dim = p0s.shape

    # pool
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_posterior,
                                    args=(z, K), pool=pool)

    _ = sampler.run_mcmc(p0s, N=256)

    pool.close()

    if continue_sampling:
        with h5py.File(output_path, 'a') as f:
            dset = f.create_dataset('chain', data=np.hstack((pre_chain, sampler.chain)))

    else:
        with h5py.File(output_path, 'w') as f:
            dset = f.create_dataset('chain', data=sampler.chain)
            dset.attrs['K'] = K

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    group_oc = parser.add_mutually_exclusive_group()
    group_oc.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                          action="store_true", help="Overwrite any existing data.")
    group_oc.add_argument("-c", "--continue", dest="_continue", default=False,
                          action="store_true", help="Continue sampling from where we left off.")

    # parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
    #                     help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("--tgas-file", dest="stacked_tgas_path", required=True,
                        type=str, help="Path to stacked TGAS data file.")
    parser.add_argument("--dist-samples", dest="distance_samples_path", required=True,
                        type=str, help="Path to HDF5 file containing distance samples.")
    parser.add_argument("--output-path", dest="output_path", default="../data/disk-z0-samples.hdf5",
                        type=str, help="Path to write output file.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)
    logger.debug("Using pool: {}".format(pool.__class__))

    # use a context manager so the prior samples file always gets deleted
    main(pool, stacked_tgas_path=args.stacked_tgas_path,
         distance_samples_path=args.distance_samples_path,
         output_path=args.output_path, # seed=args.seed,
         overwrite=args.overwrite, continue_sampling=args._continue)
