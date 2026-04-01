import numpy as np
import Lya_zelda_II as Lya


def generate_igm_transmission(w_Arr, T_p, z):  # T_p is the transmission parameter
    """
    Generates the IGM transmission curve based on a simple step function model.

    w_Arr: Wavelength array in Angstroms
    T_p: transmission parameter (float between 0 and 1)
    z: redshift
    """

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
    """
    Construct the full parameter vector (length 8)
    using free parameters from theta_free and fixed
    parameters from the config file.
    """
    full_theta = {}

    free_idx = 0
    for name in param_names:
        fp = ConfigFile["FixedParameters"][name]
        if fp["fixed"]:
            full_theta[name] = float(fp["value"])
        else:
            full_theta[name] = theta_free[free_idx]
            free_idx += 1

    return full_theta


def append_escape_fraction(chain, free_parameters, ConfigFile, ll_dict):
    """
    Calculates the escape fraction (f_esc) for the entire MCMC chain
    and appends it as a new parameter.
    """
    flat_chain = chain.reshape((-1, len(free_parameters)))
    
    def get_param_array(p_name):
        if ConfigFile['FixedParameters'][p_name]['fixed']:
            return np.full(flat_chain.shape[0], float(ConfigFile['FixedParameters'][p_name]['value']))
        else:
            return flat_chain[:, free_parameters.index(p_name)]
    
    V_arr = get_param_array('ExpV')
    LogN_arr = get_param_array('LogN')
    Tau_arr = get_param_array('Tau')
    
    Lya.funcs.Data_location = ConfigFile['GridsFolder']
    
    # Calculate f_esc for all samples vectorially
    if ConfigFile['Geometry'] == 'Thin_Shell_Cont':
        f_esc_flat = Lya.RT_f_esc('Thin_Shell', V_arr, LogN_arr, Tau_arr)
    else:
        f_esc_flat = Lya.RT_f_esc(ConfigFile['Geometry'], V_arr, LogN_arr, Tau_arr)

    nwalkers = chain.shape[0]
    nsteps = chain.shape[1]
    
    # Reshape and append to the MCMC chain
    f_esc_chain = f_esc_flat.reshape((nwalkers, nsteps, 1))
    new_chain = np.concatenate([chain, f_esc_chain], axis=2)
    
    # Update parameter lists and dictionaries
    free_parameters.append('f_esc')
    ll_dict['f_esc'] = 'f_esc'
    
    return new_chain, free_parameters, ll_dict