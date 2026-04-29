import os
import pandas as pd
import numpy as np
import yaml
import sys
import argparse
import copy

from lyafit.mcmc_routine import MCMCRoutine
from lyafit.lya_model import LyaModel
from lyafit.plotter import Plotter
from lyafit.aux_funcs import prune, build_full_theta, append_escape_fraction
from lyafit.csv_handler import CSVHandler


parser = argparse.ArgumentParser(
    description='LyaFit: MCMC Fitting of Lyman-alpha Line Profiles'
)

parser.add_argument(
    '--ConfigFile',
    required=True,
    help='Path to the configuration YAML file'
)

args = parser.parse_args()

config_file_path = args.ConfigFile

if not os.path.isfile(config_file_path):
    print(f"Configuration file '{config_file_path}' not found.")
    sys.exit(1)

with open(config_file_path, 'r') as file:
    ConfigFile = yaml.safe_load(file)

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

    # Parse Geometry as a List or String
    geometry_input = ConfigFile.get('Geometry', 'Thin_Shell_Cont')
    if isinstance(geometry_input, str):
        geometries = [geometry_input]
    elif isinstance(geometry_input, list):
        geometries = geometry_input
    else:
        print("Error: Geometry must be a string or a list of strings.")
        sys.exit(1)

    for current_geometry in geometries:
        print('\n' + 50 * '=')
        print(f'*** FITTING GEOMETRY: {current_geometry} ***')
        print(50 * '=' + '\n')

        # Create local copies for this iteration so modifications (like f_esc) don't leak
        run_config = copy.deepcopy(ConfigFile)
        run_config['Geometry'] = current_geometry
        
        # Define specific output folder and append the geometry subfolder
        run_output_folder = os.path.join(str(run_config['OutputFolder']), current_geometry)
        run_config['OutputFolder'] = run_output_folder
        
        # Reset parameters and dictionaries for each loop
        run_free_parameters = copy.deepcopy(free_parameters)
        run_ll_dict = copy.deepcopy(ll_dict)

        mcmc = MCMCRoutine(
            ndim=len(run_free_parameters),
            nwalkers=run_config['nwalkers'],
            nsteps=run_config['nsteps'],
            nthreads=run_config['Nthreads'],
            moves=run_config['MOVES'],
            mcmca=run_config['MCMCA'],
            starting_guesses=starting_guesses
        )

        FWHM_t = run_config['LSF_FWHM']
        PIX_t = run_config['PixelScale']

        lyamodel = LyaModel(
            geometry=current_geometry,
            mode=run_config['Mode'],
            free_params=run_free_parameters,
            ConfigFile=run_config,
            fwhm_t=FWHM_t,
            pix_t=PIX_t
        )

        sampler = mcmc.fit_zelda_mcmc(
            lnprob=lyamodel.lnprob,
            measured_wavelength=measured_wavelength,
            measured_flux=measured_flux,
            sigma=sigma
        )

        emcee_trace = sampler.chain[:, :, :].reshape((-1, len(run_free_parameters)))
        lnprob = sampler.lnprobability

        chain = sampler.chain.copy()
        
        if run_config.get('CalculateEscapeFraction', False):
            print(50 * '#')
            print('*** Calculating Escape Fraction ***')
            
            chain, run_free_parameters, run_ll_dict = append_escape_fraction(
                chain=chain, 
                free_parameters=run_free_parameters, 
                ConfigFile=run_config, 
                ll_dict=run_ll_dict
            )
            
            emcee_trace = chain.reshape((-1, len(run_free_parameters)))

        print(50 * '#')
        print('*** Best fit ***')

        for i in range(len(run_free_parameters)):
            if run_free_parameters[i] != 'f_esc': 
                print(run_free_parameters[i], ':', emcee_trace[np.argmax(lnprob)][i])

        theta = emcee_trace[np.argmax(lnprob)]

        print(50 * '#')
        print('*** Plotting Traces... ***')

        # Create geometry-specific directory
        os.makedirs('Results', exist_ok=True)
        os.makedirs(os.path.join('Results', run_output_folder), exist_ok=True)

        plotter = Plotter(
            chain=chain, 
            lnprob=lnprob,
            output_folder=run_output_folder,
            free_parameters=run_free_parameters,
            ll_dict=run_ll_dict,
            flux_units=run_config['FluxUnits'],
            ConfigFile=run_config
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

        samples = chain[:, nburn:, :].reshape(
            (-1, len(run_free_parameters)))
        lnprob_aux = sampler.lnprobability[:, nburn:].reshape(-1)

        print(50 * '#')
        print('*** Pruning... ***')
        
        valid_idx = np.isfinite(lnprob_aux)
        
        if np.sum(valid_idx) == 0:
            print("ERROR: All walkers returned -inf likelihood. Your parameter bounds are entirely outside the zELDA grid.")
            continue # <--- Changed to continue so it skips to the next geometry
            
        samples_valid = samples[valid_idx]
        lnprob_valid = lnprob_aux[valid_idx]

        try:
            samples, lnprob2 = prune(samples_valid, lnprob_valid)
        except Exception as e:
            print(f'Pruning failed ({e}).... Falling back to unpruned (but valid) samples.')
            samples = samples_valid
            lnprob2 = lnprob_valid

        print(50 * '#')
        print('*** Plotting Covariance... ***')

        valid_mask = np.all(np.isfinite(samples), axis=1)
        samples = samples[valid_mask]
        lnprob2 = lnprob2[valid_mask]
        
        if len(samples) == 0:
            print("ERROR: All samples contained NaNs/Infs. Check your parameter bounds.")
            continue # <--- Changed to continue so it skips to the next geometry

        plotter.plot_covariance(samples)

        print(50 * '#')
        print('*** Posterior parameters and percentiles [16,50,84]***')

        for ID in range(len(run_free_parameters)):
            pc = np.percentile(samples.T[ID], [16, 50, 84])
            print(
                run_ll_dict[run_free_parameters[ID]] + ':',
                round(pc[1], 4),
                '+/-',
                round(np.mean([pc[2] - pc[1], pc[1] - pc[0]]), 4), pc
            )

        print('*** Plotting Best Fit Over Line profile... ***')

        theta_aux = samples[np.argmax(lnprob2)]

        full_theta = build_full_theta(
            list(run_config['FixedParameters'].keys()),
            run_config,
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
            all_params=list(run_config['FixedParameters'].keys()),
            fitted_params=run_free_parameters,
            output_folder=run_output_folder,
            emcee_trace=samples,
            lnprob=lnprob2,
            ConfigFile=run_config,
            ll_dict=run_ll_dict
        )

        csv_handler.save_parameters_to_csv()

        print('*** Saving last 1000 iterations... ***')

        last_1000_chain = chain[:, -1000:, :]

        npy_path = os.path.join('Results', run_output_folder, 'last_1000_steps.npy')
        np.save(npy_path, last_1000_chain)

    print('')
    print(50 * '#')
    print('*** Done! Thank you for your patience. ***')
    print(50 * '#')
    print('')
