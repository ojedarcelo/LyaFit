import numpy as np
import Lya_zelda_II as Lya

def generate_igm_transmission(w_Arr, T_p, z):
    w_Lya = 1215.67  # Lyman-alpha wavelength in Angstroms
    w_IGM_rest_Arr = w_Arr / (1 + z)
    T_IGM_Arr = np.ones(len(w_IGM_rest_Arr))
    T_IGM_Arr[w_IGM_rest_Arr < w_Lya] = T_p
    return w_IGM_rest_Arr, T_IGM_Arr

def prune(samples, lnprob, scaler=5.0, quiet=False):
    minlnprob = lnprob.max()
    dlnprob = np.abs(lnprob - minlnprob)
    medlnprob = np.median(dlnprob)
    avglnprob = np.mean(dlnprob)
    skewlnprob = np.abs(avglnprob - medlnprob)
    rmslnprob = np.std(dlnprob)
    inliers = (dlnprob < scaler * rmslnprob)
    lnprob2 = lnprob[inliers]
    samples = samples[inliers]

    medlnprob_previous = 0.
    while skewlnprob > 0.1 * medlnprob:
        minlnprob = lnprob2.max()
        dlnprob = np.abs(lnprob2 - minlnprob)
        rmslnprob = np.std(dlnprob)
        inliers = (dlnprob < scaler * rmslnprob)
        PDFdatatmp = lnprob2[inliers]
        if len(PDFdatatmp) == len(lnprob2):
            inliers = (dlnprob < scaler / 2. * rmslnprob)
        lnprob2 = lnprob2[inliers]
        samples = samples[inliers]
        dlnprob = np.abs(lnprob2 - minlnprob)
        medlnprob = np.median(dlnprob)
        avglnprob = np.mean(dlnprob)
        skewlnprob = np.abs(avglnprob - medlnprob)
        if not quiet:
            print(medlnprob, avglnprob, skewlnprob)
        if medlnprob == medlnprob_previous:
            scaler /= 1.5
        medlnprob_previous = medlnprob
    samples = samples[lnprob2 <= minlnprob]
    lnprob2 = lnprob2[lnprob2 <= minlnprob]
    return samples, lnprob2

def build_full_theta(param_names, ConfigFile, theta_free):
    full_theta = {}
    free_idx = 0
    for name in param_names:
        if name.endswith('_2'):
            fp = ConfigFile["FixedParameters_2"][name]
        else:
            fp = ConfigFile["FixedParameters"][name]
            
        if fp["fixed"]:
            full_theta[name] = float(fp["value"])
        else:
            full_theta[name] = theta_free[free_idx]
            free_idx += 1
    return full_theta

def append_escape_fraction(chain, free_parameters, ConfigFile, ll_dict, is_two_comp=False):
    flat_chain = chain.reshape((-1, len(free_parameters)))
    nwalkers = chain.shape[0]
    nsteps = chain.shape[1]
    
    def get_param_array(p_name):
        fixed_dict = ConfigFile['FixedParameters_2'] if p_name.endswith('_2') else ConfigFile['FixedParameters']
        if fixed_dict[p_name]['fixed']:
            return np.full(flat_chain.shape[0], float(fixed_dict[p_name]['value']))
        else:
            return flat_chain[:, free_parameters.index(p_name)]
            
    Lya.funcs.Data_location = ConfigFile['GridsFolder']
    new_chain = chain.copy()

    # --- COMPONENT 1 f_esc ---
    if ConfigFile.get('CalculateEscapeFraction', False):
        V_arr = get_param_array('ExpV')
        LogN_arr = get_param_array('LogN')
        Tau_arr = get_param_array('Tau')
        
        geom = ConfigFile['Geometry']
        if geom == 'Thin_Shell_Cont': geom = 'Thin_Shell'
        
        f_esc_flat = Lya.RT_f_esc(geom, V_arr, LogN_arr, Tau_arr)
        f_esc_chain = f_esc_flat.reshape((nwalkers, nsteps, 1))
        new_chain = np.concatenate([new_chain, f_esc_chain], axis=2)
        
        free_parameters.append('f_esc')
        ll_dict['f_esc'] = 'f_esc'

    # --- COMPONENT 2 f_esc ---
    if is_two_comp and ConfigFile.get('CalculateEscapeFraction_2', False):
        V_arr_2 = get_param_array('ExpV_2')
        LogN_arr_2 = get_param_array('LogN_2')
        Tau_arr_2 = get_param_array('Tau_2')
        
        geom2 = ConfigFile['Geometry_2']
        if geom2 == 'Thin_Shell_Cont': geom2 = 'Thin_Shell'
        
        f_esc_flat_2 = Lya.RT_f_esc(geom2, V_arr_2, LogN_arr_2, Tau_arr_2)
        f_esc_chain_2 = f_esc_flat_2.reshape((nwalkers, nsteps, 1))
        new_chain = np.concatenate([new_chain, f_esc_chain_2], axis=2)
        
        free_parameters.append('f_esc_2')
        ll_dict['f_esc_2'] = 'f_esc_2'

    return new_chain, free_parameters, ll_dict