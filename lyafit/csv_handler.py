import os
import pandas as pd
import numpy as np


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
            
            # Exclude bestfit if the parameter is f_esc
            if param_name != 'f_esc':
                new_row[ll_name + '_bestfit'] = self.emcee_trace[np.argmax(self.lnprob)][i]
                
            new_row[ll_name + '_16'] = np.percentile(self.emcee_trace.T[i], 16)
            new_row[ll_name + '_50'] = np.percentile(self.emcee_trace.T[i], 50)
            new_row[ll_name + '_84'] = np.percentile(self.emcee_trace.T[i], 84)
            new_row[ll_name + '_mean'] = np.mean(self.emcee_trace.T[i])
            new_row[ll_name + '_err'] = np.std(self.emcee_trace.T[i])

        for param in self.all_params:
            if param not in self.fitted_params:
                fp = self.ConfigFile["FixedParameters"][param]
                new_row[param + '_fixed'] = fp["value"]

        df = pd.DataFrame([new_row])
        csv_path = os.path.join('Results', self.output_folder, 'results.csv')

        df.to_csv(csv_path, index=False)
        return
