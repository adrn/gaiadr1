"""
    Generate posterior samples for the distance to every TGAS star.

    Requires: astropy, emcee, h5py, numpy, schwimmbad
    (all pip-installable packages)
"""

__author__ = "adrn <adrn@princeton.edu>"

# Standard library
import os

# Third-party
from astropy.io import fits
from astropy import log as logger
import emcee
import h5py
import numpy as np
from schwimmbad import choose_pool

PARALLAX_SYSTEMATIC = 0.3 # mas

ln_twopi = np.log(2*np.pi)
ln_3 = np.log(3.)

def ln_normal(x, mu, ivar):
    return 0.5 * (np.log(ivar) - ln_twopi - (x-mu)**2 * ivar)

def ln_prior(p, d_max):
    """
    Uniform density prior up to some maximum distance, d_max.
    """
    true_plx = p[0]
    if 1/true_plx <= d_max:
        return ln_3 - 3*np.log(d_max) - 4*np.log(true_plx)
    else:
        return -np.inf

def ln_posterior(p, plx, ivar, d_max=1.):
    true_plx = p[0]
    if true_plx < 0:
        return -np.inf
    return ln_normal(true_plx, plx, ivar) + ln_prior(p, d_max)

def get_parallax_samples(parallax_obs, parallax_err, n_samples=128, d_max=1.,
                         emcee_kw=None):
    """
    """
    if emcee_kw is None:
        emcee_kw = dict()

    emcee_kw.setdefault('n_walkers', 32) # magic number
    emcee_kw.setdefault('n_burn', 256) # magic number
    n_steps = int(round(n_samples/emcee_kw['n_walkers']))

    parallax_ivar = 1/parallax_err**2
    sampler = emcee.EnsembleSampler(emcee_kw['n_walkers'], 1, lnpostfn=ln_posterior,
                                    args=(parallax_obs, parallax_ivar, d_max))

    p0 = np.random.normal(parallax_obs, 1E-2, size=(emcee_kw['n_walkers'],1))
    if np.any(1/p0 > d_max) or np.any(p0 < 0):
        p0 = np.random.normal(1/d_max, 1E-2, size=(10*emcee_kw['n_walkers'],1))
        p0 = p0[1/p0 < d_max][:emcee_kw['n_walkers'],None]

    pos,*_ = sampler.run_mcmc(p0, N=emcee_kw['n_burn'])
    sampler.reset()
    _ = sampler.run_mcmc(pos, N=n_steps)
    return sampler.flatchain[:,0]

class Worker(object):

    def __init__(self, n_samples, output_filename, d_max):
        self.n_samples = n_samples
        self.output_filename = output_filename
        self.d_max = d_max

    def work(self, i, parallax, parallax_err):
        np.random.seed(i)
        try:
            plxs = get_parallax_samples(parallax, np.sqrt(parallax_err**2 + PARALLAX_SYSTEMATIC**2),
                                        self.n_samples, self.d_max)
        except:
            logger.error("Failed for {}".format(i))
            return None

        return i, 1000/plxs # pc

    def __call__(self, task):
        i, parallax, parallax_err = task
        return self.work(i, parallax, parallax_err)

    def callback(self, result):
        if result is None:
            pass

        else:
            i,d_pc = result
            with h5py.File(self.output_filename, 'a') as f:
                f['distance'][i] = d_pc

def main(pool, stacked_tgas_path, n_samples_per_star=None, d_max=None,
         output_path=None, seed=42, overwrite=False):

    # read TGAS FITS file
    tgas = fits.getdata(stacked_tgas_path, 1)
    n_tgas = len(tgas)
    logger.debug("Loaded {} stars from stacked TGAS file.".format(n_tgas))

    if os.path.exists(output_path) and overwrite:
        os.remove(output_path)

    if not os.path.exists(output_path):
        with h5py.File(output_path, 'w') as f:
            dset = f.create_dataset('distance', dtype=np.float64,
                                    shape=(n_tgas,n_samples_per_star))
            dset.attrs['unit'] = 'pc'

    with h5py.File(output_path, 'a') as f:
        f.attrs['d_max_kpc'] = d_max

    tasks = [[k,tgas[k]['parallax'],tgas[k]['parallax_error']] for k in range(n_tgas)]

    worker = Worker(n_samples=n_samples_per_star,
                    output_filename=output_path,
                    d_max=d_max)

    for result in pool.map(worker, tasks, callback=worker.callback):
        pass

    pool.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true", help="Overwrite any existing data.")

    parser.add_argument("-n", "--n-samples", dest="n_samples_per_star", default=1024, type=int,
                        help="Number of samples to generate per star.")
    parser.add_argument("--d-max", dest="d_max", default=1., type=float,
                        help="Maximum distance in kpc.")

    # parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
    #                     help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("--tgas-file", dest="stacked_tgas_path", required=True,
                        type=str, help="Path to stacked TGAS data file.")
    parser.add_argument("--output-path", dest="output_path", default="../data/distance-samples.hdf5",
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
         n_samples_per_star=args.n_samples_per_star, d_max=args.d_max,
         output_path=args.output_path, # seed=args.seed,
         overwrite=args.overwrite)
