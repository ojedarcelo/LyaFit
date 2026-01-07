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
