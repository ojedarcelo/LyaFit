import os
import pandas as pd
import numpy as np
import yaml

from lyafit.mcmc_routine import MCMCRoutine
from lyafit.lya_model import LyaModel
from lyafit.plotter import Plotter
from lyafit.aux_funcs import prune, build_full_theta
from lyafit.csv_handler import CSVHandler

ConfigFile = yaml.safe_load(open('configLyaFit.yaml'))
ll_dict = {
    'Redshift': 'z',
    'ExpV': 'V_t',
    'LogN': 'Log_n',
    'Tau': 't_t',
    'Flux': 'F_t',
    'LogEW': 'Log_EW_t',
    'IntrinsicW': 'W_t',
    'TP': 'T_p'
}

line_df = pd.read_csv(ConfigFile['File'])

line_df.columns = line_df.columns.str.strip()

# here handle the case of inflow

measured_wavelength = line_df['w_Arr']
measured_flux = line_df['measured_flux']
sigma = line_df['sigma']

if ConfigFile['Inflow']:
    measured_flux = measured_flux[::-1]
    sigma = sigma[::-1]

nburn = int(0.5 * ConfigFile['nsteps'])
free_parameters = list()

for param in ConfigFile['FixedParameters']:
    if ConfigFile['FixedParameters'][param]['fixed']:
        print(param, ' is fixed to ', ConfigFile['FixedParameters'][param]['value'])
    else:
        free_parameters.append(param)

Bounds = [param + 'Bounds' for param in free_parameters]

for b in Bounds:
    if (ConfigFile[b][1] < ConfigFile[b][0] or
        ConfigFile[b][1] > ConfigFile[b][3] or
        ConfigFile[b][2] < ConfigFile[b][0] or
            ConfigFile[b][2] > ConfigFile[b][3]):
        print(
            'Initial Guesses outside bounds for ',
            b,
            ' :',
            ConfigFile[b])
        exit()

starting_guesses = []

for i in range(ConfigFile['nwalkers']):
    aux = [np.random.uniform(ConfigFile[param][1], ConfigFile[param][2]) for param in Bounds]
    starting_guesses.append(np.array(aux))
starting_guesses = np.array(starting_guesses)

if __name__ == '__main__':

    print(starting_guesses.shape)

    mcmc = MCMCRoutine(
        ndim=len(free_parameters),
        nwalkers=ConfigFile['nwalkers'],
        nsteps=ConfigFile['nsteps'],
        nthreads=ConfigFile['Nthreads'],
        moves=ConfigFile['MOVES'],
        mcmca=ConfigFile['MCMCA'],
        starting_guesses=starting_guesses
    )

    FWHM_t = ConfigFile['LSF_FWHM']
    PIX_t = ConfigFile['PixelScale']

    lyamodel = LyaModel(
        geometry=ConfigFile['Geometry'],
        mode=ConfigFile['Mode'],
        free_params=free_parameters,
        ConfigFile=ConfigFile,
        fwhm_t=FWHM_t,
        pix_t=PIX_t
    )

    sampler = mcmc.fit_zelda_mcmc(
        lnprob=lyamodel.lnprob,
        measured_wavelength=measured_wavelength,
        measured_flux=measured_flux,
        sigma=sigma
    )

    emcee_trace = sampler.chain[:, :, :].reshape((-1, len(free_parameters)))
    lnprob = sampler.lnprobability
    print(50 * '#')
    print('*** Best fit ***')

    for i in range(len(free_parameters)):
        print(free_parameters[i], ':', emcee_trace[np.argmax(lnprob)][i])

    theta = emcee_trace[np.argmax(lnprob)]

    print(50 * '#')
    print('*** Plotting Traces... ***')

    os.makedirs('Results', exist_ok=True)
    os.makedirs(os.path.join('Results', ConfigFile['OutputFolder']), exist_ok=True)

    plotter = Plotter(
        sampler=sampler,
        lnprob=lnprob,
        output_folder=ConfigFile['OutputFolder'],
        free_parameters=free_parameters,
        ll_dict=ll_dict,
        flux_units=ConfigFile['FluxUnits']
    )

    plotter.plot_convergence()
    plotter.plot_traces()

    print(50 * '#')
    print('*** Acceptance Fraction ***')
    af = sampler.acceptance_fraction
    af_msg = '''As a rule of thumb, the acceptance fraction (af)
                    should be between 0.2 and 0.5
            If af < 0.2 decrease the a parameter
            If af > 0.5 increase the a parameter
            '''
    print("Mean acceptance fraction:", np.mean(af))
    if np.mean(af) < 0.2 or np.mean(af) > 0.5:
        print(af_msg)

    samples = sampler.chain[:, nburn:, :].reshape(
        (-1, len(free_parameters)))
    lnprob_aux = sampler.lnprobability[:, nburn:].reshape(-1)

    print(50 * '#')
    print('*** Pruning... ***')
    try:
        samples, lnprob2 = prune(samples, lnprob_aux)
    except Exception:
        print('Prunning failed....')

    print(50 * '#')
    print('*** Plotting Covariance... ***')

    plotter.plot_covariance(samples)

    print(50 * '#')
    print('*** Posterior parameters and percentiles [16,50,84]***')

    for ID in range(len(free_parameters)):
        pc = np.percentile(samples.T[ID], [16, 50, 84])
        print(
            ll_dict[free_parameters[ID]] + ':',
            round(pc[1], 4),
            '+/-',
            round(np.mean([pc[2] - pc[1], pc[1] - pc[0]]), 4), pc
        )

    print('*** Plotting Best Fit Over Line profile... ***')

    theta_aux = samples[np.argmax(lnprob2)]

    full_theta = build_full_theta(
        list(ConfigFile['FixedParameters'].keys()),
        ConfigFile,
        theta_aux
    )

    z_t = full_theta['Redshift'] 
    V_t = full_theta['ExpV']
    log_N_t = full_theta['LogN']
    t_t = full_theta['Tau']
    F_t = full_theta['Flux']
    log_EW_t = full_theta['LogEW']
    W_t = full_theta['IntrinsicW']
    T_p = full_theta['TP']

    w_One_Arr_MCMC, f_One_Arr_MCMC, resample, info, w_IGM_rest_Arr, T_IGM_Arr = lyamodel.generate_and_resample(
        w_Arr=measured_wavelength,
        z_t=z_t,
        V_t=V_t,
        log_N_t=log_N_t,
        t_t=t_t,
        F_t=F_t,
        log_EW_t=log_EW_t,
        W_t=W_t,
        T_p=T_p
    )

    plotter.plot_best_fit(
        measured_wavelength,
        measured_flux,
        sigma,
        resample,
        z_t
    )

    print('*** Plotting IGM transmission over Best Fit... ***')

    plotter.plot_best_fit_igm(
        measured_wavelength,
        measured_flux,
        sigma,
        resample,
        z_t,
        T_IGM_Arr,
        T_p
    )

    print('*** Saving results to CSV... ***')

    csv_handler = CSVHandler(
        all_params=list(ConfigFile['FixedParameters'].keys()),
        fitted_params=free_parameters,
        output_folder=ConfigFile['OutputFolder'],
        emcee_trace=emcee_trace,
        lnprob=lnprob,
        ConfigFile=ConfigFile,
        ll_dict=ll_dict
    )

    csv_handler.save_parameters_to_csv()

    print('')
    print(50 * '#')
    print('*** Done! Thank you for your patience. ***')
    print(50 * '#')
    print('')
