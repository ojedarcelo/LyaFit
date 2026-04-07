import os
import pandas as pd
import numpy as np
from scipy import stats  # <--- NEW IMPORT

class CSVHandler:
    def __init__(self, all_params, fitted_params, output_folder, emcee_trace, lnprob, ConfigFile, ll_dict):
        self.all_params = all_params
        self.fitted_params = fitted_params
        self.output_folder = output_folder
        self.emcee_trace = emcee_trace
        self.lnprob = lnprob
        self.ConfigFile = ConfigFile
        self.ll_dict = ll_dict

    def save_parameters_to_csv(self):

        new_row = dict()

        for i in range(len(self.fitted_params)):
            param_name = self.fitted_params[i]
            ll_name = self.ll_dict[param_name]
            trace = self.emcee_trace.T[i]
            
            # --- KS Test for Uniformity ---
            if param_name == 'f_esc':
                # Escape fraction is naturally bounded between 0 and 1
                loc = 0.0
                scale = 1.0
            else:
                # Fetch absolute min and max from ConfigFile: [min, init_low, init_high, max]
                bounds = self.ConfigFile[param_name + 'Bounds']
                loc = min(trace)
                scale = max(trace) # Scale is the width of the distribution
            
            # Calculate KS statistic and p-value
            ks_stat, p_value = stats.kstest(trace, stats.uniform(loc=loc, scale=scale).cdf)
            new_row[ll_name + '_pvalue'] = p_value
            # ----------------------------------------
            
            # Exclude bestfit if the parameter is f_esc
            if param_name != 'f_esc':
                new_row[ll_name + '_bestfit'] = self.emcee_trace[np.argmax(self.lnprob)][i]
                
            new_row[ll_name + '_16'] = np.percentile(trace, 16)
            new_row[ll_name + '_50'] = np.percentile(trace, 50)
            new_row[ll_name + '_84'] = np.percentile(trace, 84)
            new_row[ll_name + '_mean'] = np.mean(trace)
            new_row[ll_name + '_err'] = np.std(trace)

        for param in self.all_params:
            if param not in self.fitted_params:
                fp = self.ConfigFile["FixedParameters"][param]
                new_row[param + '_fixed'] = fp["value"]

        df = pd.DataFrame([new_row])
        csv_path = os.path.join('Results', self.output_folder, 'results.csv')

        df.to_csv(csv_path, index=False)
        return