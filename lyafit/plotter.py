import os
import corner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


rcParams.update({'figure.autolayout': True})
sns.set_style("white", {'legend.frameon': True})
sns.set_style("ticks", {'legend.frameon': True})
sns.set_context("talk")
sns.set_palette('Dark2', desat=1)
cc = sns.color_palette()


class Plotter:
    def __init__(self, sampler, lnprob, output_folder, free_parameters, ll_dict, flux_units):
        self.sampler = sampler
        self.lnprob = lnprob
        self.results_folder_path = os.path.join('Results', output_folder)
        self.free_parameters = free_parameters
        self.ll_dict = ll_dict
        self.flux_units = flux_units

        self.LYA_WAVELENGTH = 1215.67  # Lyman-alpha wavelength in Angstroms

    def plot_convergence(self):
        x = np.array([])
        y = np.array([])
        maxlnprob = np.max(self.lnprob)
        for i in range(len(self.lnprob)):
            x = np.append(x, range(len(self.lnprob[i])))
            y = np.append(y, maxlnprob - self.lnprob[i])
        plt.figure()
        plt.hexbin(
            x[y > 0],
            y[y > 0],
            gridsize=[70, 30],
            cmap='inferno',
            bins='log',
            mincnt=1,
            yscale='log',
            linewidths=0)
        plt.ylabel('maxlnprob -lnprob')
        plt.xlabel('iteration')
        try:
            plt.xlim(min(x), max(x))
            plt.ylim(min(y), max(y))
        except Exception:
            print('Negative values in Convergence....')

        convergence_path = 'Convergence.png'
        plt.savefig(
            os.path.join(
                self.results_folder_path,
                convergence_path),
            dpi=300)
        plt.close()
        return

    def plot_traces(self):
        for ID in range(len(self.free_parameters)):
            plt.figure()
            x = np.array([])
            y = np.array([])
            for i in self.sampler.chain:
                x = np.append(x, range(len(i.T[ID])))
                y = np.append(y, i.T[ID])

            plt.figure()
            if (max(y) / min(y)) > 50 and len(y[y < 0]) < 1:
                plt.hexbin(
                    x,
                    y,
                    gridsize=[70, 30],
                    cmap='inferno',
                    bins='log',
                    mincnt=1,
                    yscale='log',
                    linewidths=0
                )
            else:
                plt.hexbin(
                    x,
                    y,
                    gridsize=[70, 30],
                    cmap='inferno',
                    bins='log',
                    mincnt=1,
                    linewidths=0
                )
            plt.ylabel(self.ll_dict[self.free_parameters[ID]])
            plt.xlabel('iteration')
            plt.xlim(min(x), max(x))
            plt.ylim(min(y), max(y))

            trace_path = self.ll_dict[self.free_parameters[ID]] + '_trace.png'
            plt.savefig(
                os.path.join(
                    self.results_folder_path,
                    trace_path),
                dpi=300)
            plt.close()

        return

    def plot_covariance(self, samples):

        ll = [self.ll_dict[p] for p in self.free_parameters]

        fig = corner.corner(
            samples,
            labels=ll,
            title_kwargs={'y': 1.05},
            title_fmt=".2f",
            use_math_text=True,
            bins=15,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            color='DarkOrange',
            hist_kwargs={'color': 'black', 'linewidth': 1.5},
            contour_kwargs={'linewidths': 1, 'colors': 'black'}
        )

        covariance_path = 'Covariance.pdf'
        fig.savefig(os.path.join(self.results_folder_path, covariance_path))
        plt.close()
        return

    def plot_best_fit(self, measured_wavelength, measured_flux, sigma, resample, z_t):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            measured_wavelength,
            measured_flux,
            c='k',
            drawstyle='steps-mid',
            linewidth=1)
        ax.plot(
            measured_wavelength,
            sigma,
            c='blue',
            drawstyle='steps-mid',
            linewidth=1.5,
            alpha=0.4,
            label='Noise')

        ax.plot(
            measured_wavelength,
            resample,
            c='g',
            label='MCMC Model',
            drawstyle='steps-mid')

        ax.set_xlabel(r'$\lambda$ (Angstrom)')
        ax.set_ylabel(
            r'Flux ({})'.format(self.flux_units))
        ax.axhline(0, color='r', ls='--')

        redshifted_wavelength = self.LYA_WAVELENGTH * (1 + z_t)
        ax.axvline(
            redshifted_wavelength,
            color='orange',
            ls='--',
            label='Best Fit Lya Wavelength\nz={}'.format(round(z_t, 3))
        )

        # ax.text(
        #     0.95,
        #     0.95,
        #     textstr,
        #     transform=ax.transAxes,
        #     fontsize=9,
        #     verticalalignment='top',
        #     horizontalalignment='right',
        #     bbox=props)

        ax.legend(loc=2, prop={'size': 10})
        plt.tight_layout()
        best_fit_path = 'BestFitOverLine.png'
        fig.savefig(
            os.path.join(self.results_folder_path, best_fit_path),
            dpi=450,
            bbox_inches='tight'
        )
        plt.close()
        return

    def plot_best_fit_igm(self, measured_wavelength, measured_flux, sigma, resample, z_t, T_IGM_Arr, T_p):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            measured_wavelength,
            measured_flux * 1. / np.amax(measured_flux),
            c='k',
            drawstyle='steps-mid',
            linewidth=1)
        ax.plot(
            measured_wavelength,
            sigma * 1. / np.amax(measured_flux),
            c='blue',
            drawstyle='steps-mid',
            linewidth=1.5,
            alpha=0.4,
            label='Noise')

        ax.plot(
            measured_wavelength,
            resample * 1. / np.amax(resample),
            c='g',
            label='MCMC Model',
            drawstyle='steps-mid')

        ax.plot(
            measured_wavelength,
            T_IGM_Arr,
            c='purple',
            label='IGM Transmission Best Fit,\nT_p = {:.3f}'.format(T_p)
        )

        ax.set_xlabel(r'$\lambda$ (Angstrom)')
        ax.set_ylabel(
            r'Flux (a. u.)')
        ax.axhline(0, color='r', ls='--')

        redshifted_wavelength = self.LYA_WAVELENGTH * (1 + z_t)
        ax.axvline(
            redshifted_wavelength,
            color='orange',
            ls='--',
            label='Best Fit Lya Wavelength\nz={}'.format(round(z_t, 3))
        )

        # ax.text(
        #     0.95,
        #     0.85,
        #     textstr,
        #     transform=ax.transAxes,
        #     fontsize=9,
        #     verticalalignment='top',
        #     horizontalalignment='right',
        #     bbox=props)

        ax.legend(loc=2, prop={'size': 10})
        plt.tight_layout()
        best_fit_path = 'BestFitOverLine_joaco_free_z_IGM.png'
        fig.savefig(
            os.path.join(self.results_folder_path, best_fit_path),
            dpi=450,
            bbox_inches='tight'
        )
        plt.close()
        return
