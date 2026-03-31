import numpy as np
import Lya_zelda_II as Lya
from lyafit.aux_funcs import generate_igm_transmission, build_full_theta

# GLOBAL CACHE: Prevents multiprocessing from pickling the grid 
# and corrupting the memory-mapped arrays across CPU cores.
_GRID_CACHE = {}

class LyaModel:
    """
    A class to load and manage different Lyman-alpha radiative transfer models from the Lya_zelda_II package.
    """

    def __init__(self, geometry, mode, free_params, ConfigFile, fwhm_t, pix_t):
        self.model_type = geometry
        self.mode = mode
        self.free_params = free_params
        self.ConfigFile = ConfigFile
        self.fwhm_t = fwhm_t
        self.pix_t = pix_t

        self.param_names = [
            "Redshift",
            "ExpV",
            "LogN",
            "Tau",
            "Flux",
            "LogEW",
            "IntrinsicW",
            "TP",
        ]

        GRIDS_LOCATION = self.ConfigFile['GridsFolder']
        Lya.funcs.Data_location = GRIDS_LOCATION

    def _get_grid(self):
        """
        Fetches the grid from the global cache. If it doesn't exist in the 
        current worker's memory, it loads it.
        """
        global _GRID_CACHE
        cache_key = (self.model_type, self.mode)
        
        if cache_key not in _GRID_CACHE:
            try:
                _GRID_CACHE[cache_key] = Lya.load_Grid_Line(self.model_type, MODE=self.mode)
            except TypeError:
                # Fallback if an older version of zELDA doesn't accept MODE for this geometry
                _GRID_CACHE[cache_key] = Lya.load_Grid_Line(self.model_type)
                
        return _GRID_CACHE[cache_key]

    def lnprior(self, theta):
        for i, pname in enumerate(self.free_params):
            bounds = self.ConfigFile[pname + 'Bounds']
            if (theta[i] < bounds[0] or theta[i] > bounds[3]):
                return -np.inf
        return 0.0

    def lnlike(self, theta, measured_wavelength, measured_flux, sigma):
        p = build_full_theta(self.param_names, self.ConfigFile, theta)
        
        # Load grid cleanly from the worker's RAM
        grid = self._get_grid()

        w_IGM_rest_Arr, T_IGM_Arr = generate_igm_transmission(
            measured_wavelength,
            T_p=p['TP'],
            z=p['Redshift']
        )

        y_model_w_Arr, y_model_f_Arr, _, info = Lya.Generate_a_real_line(
            z_t=p["Redshift"],           
            V_t=p["ExpV"],           
            log_N_t=p["LogN"],       
            t_t=p["Tau"],           
            F_t=p["Flux"],           
            log_EW_t=p["LogEW"],      
            W_t=p["IntrinsicW"],           
            PNR_t=self.ConfigFile['SNR'],            
            FWHM_t=self.fwhm_t,            
            PIX_t=self.pix_t,             
            DATA_LyaRT=grid,  
            Geometry=self.model_type,     
            T_IGM_Arr=T_IGM_Arr,    
            w_IGM_Arr=w_IGM_rest_Arr,  
            RETURN_ALL=True,
        )

        # SAFETY CATCH: If the walkers wander out of the grid bounds and zELDA 
        # returns an empty or broken array, penalize the likelihood so walkers turn back.
        if np.all(y_model_f_Arr == 0) or np.any(np.isnan(y_model_f_Arr)):
            return -np.inf

        y_model_f_Arr = np.interp(
            measured_wavelength, y_model_w_Arr, y_model_f_Arr
        )
        
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) +
                             (measured_flux - y_model_f_Arr) ** 2 /
                             sigma ** 2)

    def lnprob(self, theta, measured_wavelength, measured_flux, sigma):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        lnMeasured = self.lnlike(
            theta,
            measured_wavelength,
            measured_flux,
            sigma)
        if not np.isfinite(lnMeasured):
            return -np.inf

        return lp + lnMeasured

    def generate_and_resample(self, w_Arr, z_t, V_t, log_N_t, t_t, F_t, log_EW_t, W_t, T_p):
        w_IGM_rest_Arr, T_IGM_Arr = generate_igm_transmission(
            w_Arr, T_p=T_p, z=z_t)

        grid = self._get_grid()

        w_One_Arr_MCMC, f_One_Arr_MCMC, _, info = Lya.Generate_a_real_line(
            z_t=z_t,
            V_t=V_t,
            log_N_t=log_N_t,
            t_t=t_t,
            F_t=F_t,
            log_EW_t=log_EW_t,
            W_t=W_t,
            PNR_t=self.ConfigFile['SNR'],
            FWHM_t=self.fwhm_t,
            PIX_t=self.pix_t,
            DATA_LyaRT=grid,
            Geometry=self.model_type,
            T_IGM_Arr=T_IGM_Arr,
            w_IGM_Arr=w_IGM_rest_Arr,
            RETURN_ALL=True
        )

        resample = np.interp(w_Arr, w_One_Arr_MCMC, f_One_Arr_MCMC)
        return w_One_Arr_MCMC, f_One_Arr_MCMC, resample, info, w_IGM_rest_Arr, T_IGM_Arr