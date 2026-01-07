import Lya_zelda_II as Lya
from aux_funcs import generate_igm_transmission


class LyaModel:
    """
    A class to load and manage different Lyman-alpha radiative transfer models from the Lya_zelda_II package.
    """

    def __init__(self, geometry, mode, **kwargs):
        self.model_type = geometry
        self.model = self.load_model()
        self.mode = mode
        

    def lnlike(self, theta, measured_wavelength, sigma):
        LyaRT_Grid = Lya.load_Grid_Line(self.geometry, MODE=self.mode)

        if self.IGM:
            w_IGM_rest_Arr, T_IGM_Arr = generate_igm_transmission(
                measured_wavelength, T_p=theta[7], z=theta[0])
            


