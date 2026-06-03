import numpy as np
import Lya_zelda_II as Lya
from lyafit.aux_funcs import generate_igm_transmission, build_full_theta, gaussian

_GRID_CACHE = {}

class LyaModel:
    def __init__(self, geometry, mode, free_params, ConfigFile, fwhm_t, pix_t,
                 is_two_comp=False, geometry_2=None, mode_2=None, gaussian_component=False):
        self.model_type = geometry
        self.mode = mode
        self.free_params = free_params
        self.ConfigFile = ConfigFile
        self.fwhm_t = fwhm_t
        self.pix_t = pix_t
        self.is_two_comp = is_two_comp
        self.model_type_2 = geometry_2
        self.mode_2 = mode_2
        self.gaussian_component = gaussian_component

        self.param_names = ["Redshift", "ExpV", "LogN", "Tau", "Flux", "LogEW", "IntrinsicW", "TP"]
        self.gaussian_param_names = ["GaussianCenter", "GaussianFWHM", "GaussianAmplitude"] if self.gaussian_component else []
        
        self.all_param_names = list(self.param_names) + self.gaussian_param_names
        if self.is_two_comp:
            self.all_param_names += [p + "_2" for p in self.param_names]

        GRIDS_LOCATION = self.ConfigFile['GridsFolder']
        Lya.funcs.Data_location = GRIDS_LOCATION

    def _get_grid(self, geom, mode):
        global _GRID_CACHE
        cache_key = (geom, mode)
        
        if cache_key not in _GRID_CACHE:
            try:
                _GRID_CACHE[cache_key] = Lya.load_Grid_Line(geom, MODE=mode)
            except TypeError:
                _GRID_CACHE[cache_key] = Lya.load_Grid_Line(geom)
                
        return _GRID_CACHE[cache_key]

    def lnprior(self, theta):
        for i, pname in enumerate(self.free_params):
            if pname.startswith('f_esc'): 
                continue 
                
            if pname.endswith('_2'):
                base = pname[:-2]
                bounds = self.ConfigFile[base + 'Bounds_2']
            else:
                bounds = self.ConfigFile[pname + 'Bounds']
                
            if (theta[i] < bounds[0] or theta[i] > bounds[3]):
                return -np.inf
        return 0.0

    def lnlike(self, theta, measured_wavelength, measured_flux, sigma):
        p = build_full_theta(self.all_param_names, self.ConfigFile, theta)
        
        grid_1 = self._get_grid(self.model_type, self.mode)

        w_IGM_rest_Arr_1, T_IGM_Arr_1 = generate_igm_transmission(
            measured_wavelength, T_p=p['TP'], z=p['Redshift']
        )

        y_model_w_Arr_1, y_model_f_Arr_1, _, _ = Lya.Generate_a_real_line(
            z_t=p["Redshift"], V_t=p["ExpV"], log_N_t=p["LogN"], t_t=p["Tau"],
            F_t=p["Flux"], log_EW_t=p["LogEW"], W_t=p["IntrinsicW"],
            PNR_t=self.ConfigFile['SNR'], FWHM_t=self.fwhm_t, PIX_t=self.pix_t,
            DATA_LyaRT=grid_1, Geometry=self.model_type,
            T_IGM_Arr=T_IGM_Arr_1, w_IGM_Arr=w_IGM_rest_Arr_1, RETURN_ALL=True,
        )

        if np.all(y_model_f_Arr_1 == 0) or np.any(np.isnan(y_model_f_Arr_1)):
            return -np.inf

        y_model_f_tot = np.interp(measured_wavelength, y_model_w_Arr_1, y_model_f_Arr_1)

        if self.gaussian_component:
            gauss_comp = gaussian(
                measured_wavelength,
                p['GaussianCenter'],
                p['GaussianFWHM'],
                p['GaussianAmplitude']
            )
            y_model_f_tot += gauss_comp
        
        if self.is_two_comp:
            grid_2 = self._get_grid(self.model_type_2, self.mode_2)
            
            w_IGM_rest_Arr_2, T_IGM_Arr_2 = generate_igm_transmission(
                measured_wavelength, T_p=p['TP_2'], z=p['Redshift_2']
            )

            y_model_w_Arr_2, y_model_f_Arr_2, _, _ = Lya.Generate_a_real_line(
                z_t=p["Redshift_2"], V_t=p["ExpV_2"], log_N_t=p["LogN_2"], t_t=p["Tau_2"],
                F_t=p["Flux_2"], log_EW_t=p["LogEW_2"], W_t=p["IntrinsicW_2"],
                PNR_t=self.ConfigFile['SNR'], FWHM_t=self.fwhm_t, PIX_t=self.pix_t,
                DATA_LyaRT=grid_2, Geometry=self.model_type_2,
                T_IGM_Arr=T_IGM_Arr_2, w_IGM_Arr=w_IGM_rest_Arr_2, RETURN_ALL=True,
            )

            if np.all(y_model_f_Arr_2 == 0) or np.any(np.isnan(y_model_f_Arr_2)):
                return -np.inf

            y_model_f_interp_2 = np.interp(measured_wavelength, y_model_w_Arr_2, y_model_f_Arr_2)
            y_model_f_tot += y_model_f_interp_2

        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) +
                             (measured_flux - y_model_f_tot) ** 2 / sigma ** 2)

    def lnprob(self, theta, measured_wavelength, measured_flux, sigma):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        lnMeasured = self.lnlike(theta, measured_wavelength, measured_flux, sigma)
        if not np.isfinite(lnMeasured):
            return -np.inf

        return lp + lnMeasured

    def generate_and_resample(self, w_Arr, theta_dict):
        p = theta_dict
        
        grid_1 = self._get_grid(self.model_type, self.mode)
        w_IGM_rest_Arr_1, T_IGM_Arr_1 = generate_igm_transmission(w_Arr, T_p=p['TP'], z=p['Redshift'])
        
        w_One_Arr_MCMC_1, f_One_Arr_MCMC_1, _, info_1 = Lya.Generate_a_real_line(
            z_t=p["Redshift"], V_t=p["ExpV"], log_N_t=p["LogN"], t_t=p["Tau"],
            F_t=p["Flux"], log_EW_t=p["LogEW"], W_t=p["IntrinsicW"],
            PNR_t=self.ConfigFile['SNR'], FWHM_t=self.fwhm_t, PIX_t=self.pix_t,
            DATA_LyaRT=grid_1, Geometry=self.model_type,
            T_IGM_Arr=T_IGM_Arr_1, w_IGM_Arr=w_IGM_rest_Arr_1, RETURN_ALL=True
        )
        
        resample_1 = np.interp(w_Arr, w_One_Arr_MCMC_1, f_One_Arr_MCMC_1)
        
        models_dict = {
            'w_1': w_One_Arr_MCMC_1, 'f_1': f_One_Arr_MCMC_1, 'resample_1': resample_1,
            'w_IGM_1': w_IGM_rest_Arr_1, 'T_IGM_1': T_IGM_Arr_1, 'info_1': info_1,
            'resample_tot': resample_1.copy()
        }

        if self.gaussian_component:
            gauss_comp = gaussian(
                w_Arr,
                p['GaussianCenter'],
                p['GaussianFWHM'],
                p['GaussianAmplitude']
            )
            models_dict['gaussian'] = gauss_comp
            models_dict['resample_tot'] += gauss_comp

        if self.is_two_comp:
            grid_2 = self._get_grid(self.model_type_2, self.mode_2)
            w_IGM_rest_Arr_2, T_IGM_Arr_2 = generate_igm_transmission(w_Arr, T_p=p['TP_2'], z=p['Redshift_2'])
            
            w_One_Arr_MCMC_2, f_One_Arr_MCMC_2, _, info_2 = Lya.Generate_a_real_line(
                z_t=p["Redshift_2"], V_t=p["ExpV_2"], log_N_t=p["LogN_2"], t_t=p["Tau_2"],
                F_t=p["Flux_2"], log_EW_t=p["LogEW_2"], W_t=p["IntrinsicW_2"],
                PNR_t=self.ConfigFile['SNR'], FWHM_t=self.fwhm_t, PIX_t=self.pix_t,
                DATA_LyaRT=grid_2, Geometry=self.model_type_2,
                T_IGM_Arr=T_IGM_Arr_2, w_IGM_Arr=w_IGM_rest_Arr_2, RETURN_ALL=True
            )
            
            resample_2 = np.interp(w_Arr, w_One_Arr_MCMC_2, f_One_Arr_MCMC_2)
            
            models_dict.update({
                'w_2': w_One_Arr_MCMC_2, 'f_2': f_One_Arr_MCMC_2, 'resample_2': resample_2,
                'w_IGM_2': w_IGM_rest_Arr_2, 'T_IGM_2': T_IGM_Arr_2, 'info_2': info_2
            })
            
            models_dict['resample_tot'] = resample_1 + resample_2

        return models_dict