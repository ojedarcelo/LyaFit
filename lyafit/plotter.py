import os
import corner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.constants as const
from scipy import stats
from matplotlib import rcParams

from lyafit.aux_funcs import w2v, v2w


rcParams.update({'figure.autolayout': True})
sns.set_style("white", {'legend.frameon': True})
sns.set_style("ticks", {'legend.frameon': True})
sns.set_context("talk")
sns.set_palette('Dark2', desat=1)
cc = sns.color_palette()


class Plotter:

    def __init__(
            self, chain, lnprob, output_folder, free_parameters, ll_dict, flux_units, ConfigFile, is_two_comp=False
    ):
        self.chain = chain
        self.lnprob = lnprob
        self.results_folder_path = os.path.join('Results', output_folder)
        self.free_parameters = free_parameters
        self.ll_dict = ll_dict
        self.flux_units = flux_units
        self.ConfigFile = ConfigFile
        self.is_two_comp = is_two_comp

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
        plt.savefig(os.path.join(self.results_folder_path, convergence_path), dpi=300)
        plt.close()
        return

    def plot_traces(self):
        for ID in range(len(self.free_parameters)):
            plt.figure()
            x = np.array([])
            y = np.array([])
            for i in self.chain:
                x = np.append(x, range(len(i.T[ID])))
                y = np.append(y, i.T[ID])

            # --- NEW CODE: Safely ignore NaNs ---
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            # ------------------------------------

            plt.figure()
            if y_min > 0 and (y_max / y_min) > 50:
                plt.hexbin(
                    x, y, gridsize=[70, 30], cmap='inferno',
                    bins='log', mincnt=1, yscale='log', linewidths=0
                )
            else:
                plt.hexbin(
                    x, y, gridsize=[70, 30], cmap='inferno',
                    bins='log', mincnt=1, linewidths=0
                )
            plt.ylabel(self.ll_dict[self.free_parameters[ID]])
            plt.xlabel('iteration')
            
            # --- Apply safe limits ---
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            # -------------------------

            trace_path = self.ll_dict[self.free_parameters[ID]] + '_trace.png'
            plt.savefig(os.path.join(self.results_folder_path, trace_path), dpi=300)
            plt.close()
        return

    def plot_covariance(self, samples, comp_suffix=''):
        ll_with_pvalues = []
        comp_indices = []

        for i, p_name in enumerate(self.free_parameters):
            is_comp_2 = p_name.endswith('_2') or p_name == 'f_esc_2'

            # Split the logic so we only graph Component 1 OR Component 2
            if comp_suffix == '_2' and not is_comp_2:
                continue
            if comp_suffix == '' and is_comp_2:
                continue

            comp_indices.append(i)
            trace = samples.T[i]
            ll_name = self.ll_dict[p_name]

            if p_name.startswith('f_esc'):
                loc, scale = 0.0, 1.0
            else:
                if p_name.endswith('_2'):
                    base = p_name[:-2]
                    bounds = self.ConfigFile[base + 'Bounds_2']
                else:
                    bounds = self.ConfigFile[p_name + 'Bounds']
                loc, scale = bounds[0], bounds[3] - bounds[0]

            _, p_value = stats.kstest(trace, stats.uniform(loc=loc, scale=scale).cdf)

            if p_value < 0.001:
                p_str = "p < 0.001"
            else:
                p_str = f"p={p_value:.3f}"

            ll_with_pvalues.append(f"{ll_name}\n({p_str})")

        comp_samples = samples[:, comp_indices]
        ndim = len(comp_indices)
        custom_fig = plt.figure(figsize=(ndim * 3, ndim * 3))

        fig = corner.corner(
            comp_samples,
            fig=custom_fig,
            labels=ll_with_pvalues,
            label_kwargs={'labelpad': 30},
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

        covariance_path = f'Covariance{comp_suffix}.pdf'
        fig.savefig(
            os.path.join(self.results_folder_path, covariance_path),
            bbox_inches='tight',
            pad_inches=0.2
        )
        plt.close()
        return

    def plot_best_fit(self, measured_wavelength, measured_flux, sigma, models_dict, full_theta, is_two_comp):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(measured_wavelength, measured_flux, c='k', drawstyle='steps-mid', linewidth=1)
        ax.plot(measured_wavelength, sigma, c='blue', drawstyle='steps-mid', linewidth=1.5, alpha=0.4, label='Noise')

        if is_two_comp:
            geom1 = self.ConfigFile.get('Geometry', 'Model 1')
            geom2 = self.ConfigFile.get('Geometry_2', 'Model 2')

            ax.plot(measured_wavelength, models_dict['resample_1'], c='r',
                    label=f'Component 1: {geom1}', drawstyle='steps-mid', alpha=0.6)
            ax.plot(measured_wavelength, models_dict['resample_2'], c='b',
                    label=f'Component 2: {geom2}', drawstyle='steps-mid', alpha=0.6)
            ax.plot(measured_wavelength, models_dict['resample_tot'], c='g',
                    label='Full Model', drawstyle='steps-mid', linewidth=2)
        else:
            ax.plot(measured_wavelength, models_dict['resample_tot'], c='g',
                    label='MCMC Model', drawstyle='steps-mid')

        ax.set_xlabel(r'$\lambda$ (Angstrom)', fontsize=16)
        ax.set_ylabel(r'Flux ({})'.format(self.flux_units), fontsize=16)
        ax.axhline(0, color='r', ls='--')

        z_t_1 = full_theta['Redshift']
        lambda_0 = self.LYA_WAVELENGTH * (1 + z_t_1)
        c_kms = const.c.to('km/s').value

        secax = ax.secondary_xaxis(
            'top',
            functions=(lambda w: w2v(w, lambda_0, c_kms), lambda v: v2w(v, lambda_0, c_kms))
        )
        secax.set_xlabel('Velocity (km/s)', labelpad=10, fontsize=16)

        is_z_fixed_1 = self.ConfigFile.get('FixedParameters', {}).get('Redshift', {}).get('fixed', False)

        z_label_1 = f'Sys Lya z={round(z_t_1, 3)}' if is_z_fixed_1 else f'Best Fit Lya z={round(z_t_1, 3)}'
        ax.axvline(self.LYA_WAVELENGTH * (1 + z_t_1), color='orange', ls='--', label=z_label_1)

        if is_two_comp:
            z_t_2 = full_theta['Redshift_2']
            is_z_fixed_2 = self.ConfigFile.get('FixedParameters_2', {}).get('Redshift_2', {}).get('fixed', False)
            z_label_2 = f'Sys Lya 2 z={round(z_t_2, 3)}' if is_z_fixed_2 else f'Best Fit Lya 2 z={round(z_t_2, 3)}'
            ax.axvline(self.LYA_WAVELENGTH * (1 + z_t_2), color='magenta', ls='--', label=z_label_2)

        ax.tick_params(axis='both', which='major', labelsize=12)
        secax.tick_params(axis='x', which='major', labelsize=12)

        ax.legend(loc=2, prop={'size': 10})
        plt.tight_layout()
        best_fit_path = 'BestFitOverLine.png'
        fig.savefig(os.path.join(self.results_folder_path, best_fit_path), dpi=450, bbox_inches='tight')
        plt.close()
        return

    def plot_best_fit_igm(self, measured_wavelength, measured_flux, sigma, models_dict, full_theta, is_two_comp):
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            measured_wavelength, measured_flux * 1. / np.amax(measured_flux),
            c='k',
            drawstyle='steps-mid',
            linewidth=1
        )

        ax.plot(
            measured_wavelength, sigma * 1. / np.amax(measured_flux),
            c='blue',
            drawstyle='steps-mid',
            linewidth=1.5,
            alpha=0.4,
            label='Noise'
        )

        if is_two_comp:
            geom1 = self.ConfigFile.get('Geometry', 'Model 1')
            geom2 = self.ConfigFile.get('Geometry_2', 'Model 2')

            resample_norm = np.amax(models_dict['resample_tot'])

            ax.plot(measured_wavelength, models_dict['resample_1'] / resample_norm, c='r',
                    label=f'Component 1: {geom1}', drawstyle='steps-mid', alpha=0.6)
            ax.plot(measured_wavelength, models_dict['resample_2'] / resample_norm, c='b',
                    label=f'Component 2: {geom2}', drawstyle='steps-mid', alpha=0.6)
            ax.plot(measured_wavelength, models_dict['resample_tot'] / resample_norm, c='g',
                    label='Full Model', drawstyle='steps-mid', linewidth=2)

            ax.plot(measured_wavelength, models_dict['T_IGM_1'], c='purple',
                    label='IGM 1, T_p = {:.3f}'.format(full_theta['TP']))
            ax.plot(measured_wavelength, models_dict['T_IGM_2'], c='brown',
                    label='IGM 2, T_p = {:.3f}'.format(full_theta['TP_2']))
        else:
            ax.plot(measured_wavelength, models_dict['resample_tot'] * 1. / np.amax(models_dict['resample_tot']),
                    c='g', label='MCMC Model', drawstyle='steps-mid')
            ax.plot(measured_wavelength, models_dict['T_IGM_1'], c='purple',
                    label='IGM Transmission, T_p = {:.3f}'.format(full_theta['TP']))

        ax.set_xlabel(r'$\lambda$ (Angstrom)', fontsize=16)
        ax.set_ylabel(r'Flux (a. u.)', fontsize=16)
        ax.axhline(0, color='r', ls='--')

        z_t_1 = full_theta['Redshift']
        lambda_0 = self.LYA_WAVELENGTH * (1 + z_t_1)
        c_kms = const.c.to('km/s').value

        secax = ax.secondary_xaxis(
            'top',
            functions=(lambda w: w2v(w, lambda_0, c_kms), lambda v: v2w(v, lambda_0, c_kms))
        )
        secax.set_xlabel('Velocity (km/s)', labelpad=10, fontsize=16)

        is_z_fixed_1 = self.ConfigFile.get('FixedParameters', {}).get('Redshift', {}).get('fixed', False)

        z_label_1 = f'Sys Lya z={round(z_t_1, 3)}' if is_z_fixed_1 else f'Best Fit Lya z={round(z_t_1, 3)}'
        ax.axvline(self.LYA_WAVELENGTH * (1 + z_t_1), color='orange', ls='--', label=z_label_1)

        if is_two_comp:
            z_t_2 = full_theta['Redshift_2']
            is_z_fixed_2 = self.ConfigFile.get('FixedParameters_2', {}).get('Redshift_2', {}).get('fixed', False)
            z_label_2 = f'Sys Lya 2 z={round(z_t_2, 3)}' if is_z_fixed_2 else f'Best Fit Lya 2 z={round(z_t_2, 3)}'
            ax.axvline(self.LYA_WAVELENGTH * (1 + z_t_2), color='magenta', ls='--', label=z_label_2)

        ax.tick_params(axis='both', which='major', labelsize=12)
        secax.tick_params(axis='x', which='major', labelsize=12)

        ax.legend(loc=2, prop={'size': 10})
        plt.tight_layout()
        best_fit_path = 'BestFitOverLine_IGM.png'
        fig.savefig(os.path.join(self.results_folder_path, best_fit_path), dpi=450, bbox_inches='tight')
        plt.close()
        return
