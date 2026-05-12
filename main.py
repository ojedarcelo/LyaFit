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

# --- Detect 2-Component Mode ---
keys_2 = [
    'RedshiftBounds_2', 'ExpVBounds_2', 'LogNBounds_2', 'TauBounds_2',
    'FluxBounds_2', 'LogEWBounds_2', 'IntrinsicWBounds_2', 'TPBounds_2',
    'CalculateEscapeFraction_2', 'Geometry_2', 'Mode_2', 'FixedParameters_2'
]

present_keys_2 = [k for k in keys_2 if k in ConfigFile]
is_two_comp = len(present_keys_2) > 0

if is_two_comp:
    if len(present_keys_2) != len(keys_2):
        missing = [k for k in keys_2 if k not in ConfigFile]
        print(f"Error: Second model parameter block is semi-commented. Missing keys: {missing}")
        sys.exit(1)
        
    ll_dict.update({
        'Redshift_2': 'z_2', 'ExpV_2': 'V_t_2', 'LogN_2': 'Log_n_2', 'Tau_2': 't_t_2',
        'Flux_2': 'F_t_2', 'LogEW_2': 'Log_EW_t_2', 'IntrinsicW_2': 'W_t_2', 'TP_2': 'T_p_2'
    })

line_df = pd.read_csv(ConfigFile['File'])
line_df.columns = line_df.columns.str.strip()

measured_wavelength = line_df['w_Arr'].values
measured_flux = line_df['measured_flux'].values
sigma = line_df['sigma'].values

# Inflow is a global data property, applied to the entire spectrum
if ConfigFile.get('Inflow', False):
    measured_flux = measured_flux[::-1]
    sigma = sigma[::-1]

nburn = int(0.5 * ConfigFile['nsteps'])
free_parameters = list()

for param in ConfigFile['FixedParameters']:
    if ConfigFile['FixedParameters'][param]['fixed']:
        print(param, ' is fixed to ', ConfigFile['FixedParameters'][param]['value'])
    else:
        free_parameters.append(param)

if is_two_comp:
    for param in ConfigFile['FixedParameters_2']:
        if ConfigFile['FixedParameters_2'][param]['fixed']:
            print(param, ' is fixed to ', ConfigFile['FixedParameters_2'][param]['value'])
        else:
            free_parameters.append(param)

Bounds_dict = {}
for param in free_parameters:
    if param.endswith('_2'):
        base = param[:-2]
        Bounds_dict[param] = ConfigFile[base + 'Bounds_2']
    else:
        Bounds_dict[param] = ConfigFile[param + 'Bounds']

for param, b in Bounds_dict.items():
    if (b[1] < b[0] or b[1] > b[3] or b[2] < b[0] or b[2] > b[3]):
        print(f'Initial Guesses outside bounds for {param}: {b}')
        exit()

starting_guesses = []
for i in range(ConfigFile['nwalkers']):
    aux = [np.random.uniform(Bounds_dict[param][1], Bounds_dict[param][2]) for param in free_parameters]
    starting_guesses.append(np.array(aux))
starting_guesses = np.array(starting_guesses)


if __name__ == '__main__':

    print(starting_guesses.shape)

    geometry_input = ConfigFile.get('Geometry', 'Thin_Shell_Cont')
    if isinstance(geometry_input, str):
        geometries = [geometry_input]
    elif isinstance(geometry_input, list):
        geometries = geometry_input
    else:
        print("Error: Geometry must be a string or a list of strings.")
        sys.exit(1)

    if is_two_comp:
        geometry_input_2 = ConfigFile.get('Geometry_2')
        if isinstance(geometry_input_2, str):
            geometries_2 = [geometry_input_2]
        elif isinstance(geometry_input_2, list):
            geometries_2 = geometry_input_2
        else:
            print("Error: Geometry_2 must be a string or a list of strings.")
            sys.exit(1)
            
        if len(geometries) > 1 or len(geometries_2) > 1:
            print("Error: In 2-component mode, Geometry and Geometry_2 must evaluate to a single string.")
            sys.exit(1)
        loop_iterable = [(geometries[0], geometries_2[0])]
    else:
        loop_iterable = [(geom, None) for geom in geometries]


    for current_geometry, current_geometry_2 in loop_iterable:
        print('\n' + 50 * '=')
        if is_two_comp:
            print(f'*** FITTING GEOMETRY: {current_geometry} + {current_geometry_2} ***')
        else:
            print(f'*** FITTING GEOMETRY: {current_geometry} ***')
        print(50 * '=' + '\n')

        run_config = copy.deepcopy(ConfigFile)
        run_config['Geometry'] = current_geometry
        
        if is_two_comp:
            run_config['Geometry_2'] = current_geometry_2
            run_output_folder = os.path.join(str(run_config['OutputFolder']), f"{current_geometry}_{current_geometry_2}")
        else:
            run_output_folder = os.path.join(str(run_config['OutputFolder']), current_geometry)
            
        run_config['OutputFolder'] = run_output_folder
        
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
            mode=run_config.get('Mode', 'Light'),
            free_params=run_free_parameters,
            ConfigFile=run_config,
            fwhm_t=FWHM_t,
            pix_t=PIX_t,
            is_two_comp=is_two_comp,
            geometry_2=current_geometry_2,
            mode_2=run_config.get('Mode_2')
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
        
        if run_config.get('CalculateEscapeFraction', False) or (is_two_comp and run_config.get('CalculateEscapeFraction_2', False)):
            print(50 * '#')
            print('*** Calculating Escape Fraction(s) ***')
            chain, run_free_parameters, run_ll_dict = append_escape_fraction(
                chain=chain, 
                free_parameters=run_free_parameters, 
                ConfigFile=run_config, 
                ll_dict=run_ll_dict,
                is_two_comp=is_two_comp
            )
            emcee_trace = chain.reshape((-1, len(run_free_parameters)))

        print(50 * '#')
        print('*** Best fit ***')

        for i in range(len(run_free_parameters)):
            if not run_free_parameters[i].startswith('f_esc'): 
                print(run_free_parameters[i], ':', emcee_trace[np.argmax(lnprob)][i])

        theta = emcee_trace[np.argmax(lnprob)]

        print(50 * '#')
        print('*** Plotting Traces... ***')

        os.makedirs('Results', exist_ok=True)
        os.makedirs(os.path.join('Results', run_output_folder), exist_ok=True)

        plotter = Plotter(
            chain=chain, 
            lnprob=lnprob,
            output_folder=run_output_folder,
            free_parameters=run_free_parameters,
            ll_dict=run_ll_dict,
            flux_units=run_config['FluxUnits'],
            ConfigFile=run_config,
            is_two_comp=is_two_comp
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
            print("ERROR: All walkers returned -inf likelihood.")
            continue 
            
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
            print("ERROR: All samples contained NaNs/Infs.")
            continue 

        # Plot Component 1
        plotter.plot_covariance(samples)
        
        # Plot Component 2 if active
        if is_two_comp:
            plotter.plot_covariance(samples, comp_suffix='_2')

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
        
        all_params_keys = list(run_config['FixedParameters'].keys())
        if is_two_comp:
            all_params_keys += list(run_config['FixedParameters_2'].keys())

        full_theta = build_full_theta(
            all_params_keys,
            run_config,
            theta_aux
        )

        models_dict = lyamodel.generate_and_resample(
            w_Arr=measured_wavelength,
            theta_dict=full_theta
        )

        plotter.plot_best_fit(
            measured_wavelength,
            measured_flux,
            sigma,
            models_dict,
            full_theta,
            is_two_comp
        )

        print('*** Plotting IGM transmission over Best Fit... ***')

        plotter.plot_best_fit_igm(
            measured_wavelength,
            measured_flux,
            sigma,
            models_dict,
            full_theta,
            is_two_comp
        )

        print('*** Saving results and fitted spectrum to CSV... ***')

        csv_handler = CSVHandler(
            all_params=all_params_keys,
            fitted_params=run_free_parameters,
            output_folder=run_output_folder,
            emcee_trace=samples,
            lnprob=lnprob2,
            ConfigFile=run_config,
            ll_dict=run_ll_dict
        )

        # Save standard parameters
        csv_handler.save_parameters_to_csv()
        
        # Save the new fitted arrays
        csv_handler.save_fitted_spectrum_to_csv(
            w_arr=measured_wavelength,
            models_dict=models_dict,
            is_two_comp=is_two_comp
        )

        print('*** Saving last 1000 iterations... ***')
        last_1000_chain = chain[:, -1000:, :]
        npy_path = os.path.join('Results', run_output_folder, 'last_1000_steps.npy')
        np.save(npy_path, last_1000_chain)

    print('')
    print(50 * '#')
    print('*** Done! Thank you for your patience. ***')
    print(50 * '#')
    print('')