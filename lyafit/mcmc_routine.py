import emcee
import time
import numpy as np
from multiprocessing import Pool


class MCMCRoutine:
    def __init__(self, ndim, nwalkers, nsteps, nthreads, moves, mcmca, starting_guesses):
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nthreads = nthreads
        self.moves = moves
        self.mcmca = mcmca
        self.starting_guesses = starting_guesses

    def fit_zelda_mcmc(self, lnprob, measured_wavelength, measured_flux, sigma):

        print('Number of iterations:', self.ndim * self.nwalkers * self.nsteps)

        with Pool(self.nthreads) as pool:
            if self.moves == 'Stretch':
                sampler = emcee.EnsembleSampler(
                    self.nwalkers,
                    self.ndim,
                    lnprob,
                    args=[measured_wavelength, measured_flux, sigma],
                    pool=pool,
                    moves=emcee.moves.StretchMove(self.mcmca)
                )
            elif self.moves == 'KDE':
                sampler = emcee.EnsembleSampler(
                    self.nwalkers,
                    self.ndim,
                    lnprob,
                    args=[measured_wavelength, measured_flux, sigma],
                    pool=pool,
                    moves=emcee.moves.KDEMove()
                )
            else:
                sampler = emcee.EnsembleSampler(
                    self.nwalkers,
                    self.ndim,
                    lnprob,
                    args=[measured_wavelength, measured_flux, sigma],
                    pool=pool
                )

            currenttime = time.time()
            Step = 1
            for pos, prob, state in sampler.sample(
                    self.starting_guesses, iterations=self.nsteps):
                print('Step:', Step, '/', self.nsteps)
                print("Mean acceptance fraction: %f" % (np.mean(
                    sampler.acceptance_fraction
                )))
                print("Mean lnprob and Max lnprob values: %f %f" % (
                    np.mean(prob), np.max(prob)
                ))
                print(
                    "Time to run previous set of walkers (seconds): %f" %
                    (time.time() - currenttime))
                currenttime = time.time()
                Step += 1

        print('*** Done Fitting... ***')

        return sampler
