import numpy as np


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
            full_theta[name] = fp["value"]
        else:
            full_theta[name] = theta_free[free_idx]
            free_idx += 1

    return full_theta
