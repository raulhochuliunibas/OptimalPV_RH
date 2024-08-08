import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from collections import OrderedDict
from numpy.polynomial.polynomial import Polynomial


sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# INVESTMENT COSTS
# ------------------------------------------------------------------------------------------------------

def attach_pv_cost(
        dataagg_settings_def, ):        
    """
    Function to create assumed cost df for PV installation (by total and relative size in kW)
    Source: https://www.energieschweiz.ch/tools/solarrechner/
    """ 
    # setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']


    print_to_logfile(f'run function: attach_pv_cost.py', log_file_name_def= log_file_name_def)


    # CREATE COST DF AND FUNCTIONS ================================================================
    if True:
        conversion_m2_to_kw = 0.1  # A 1m2 area can fit 0.1 kWp of PV Panels
        installation_cost_dict = {
            "on_roof_installation_cost_pkW": {
                2:   4636,
                3:   3984,
                5:   3373,
                10:  2735,
                15:  2420,
                20:  2219,
                30:  1967,
                50:  1710,
                75:  1552,
                100: 1463,
                125: 1406,
                150: 1365
            },
            "on_roof_installation_cost_total": {
                2:   9272,
                3:   11952,
                5:   16863,
                10:  27353,
                15:  36304,
                20:  44370,
                30:  59009,
                50:  85478,
                75:  116420,
                100: 146349,
                125: 175748,
                150: 204816
            },}
        
        installation_cost_df = pd.DataFrame({
            'kw': list(installation_cost_dict['on_roof_installation_cost_pkW'].keys()),
            'chf_pkW': list(installation_cost_dict['on_roof_installation_cost_pkW'].values()),
            'chf_total': list(installation_cost_dict['on_roof_installation_cost_total'].values())
        })
        installation_cost_df.reset_index(inplace=True)

        # define intrapolation functions for cost structure -------------------
        # chf_pkW
        def func_chf_pkW(x, a, b):
            return a + b / x
        params_pkW, covar = curve_fit(func_chf_pkW, installation_cost_df['kw'], installation_cost_df['chf_pkW'])
        # createa a function that takes a kw value and returns the cost per kW
        estim_instcost_chfpkW = lambda x: func_chf_pkW(x, *params_pkW)
        print_to_logfile(f'params_pkW: {params_pkW}', log_file_name_def)
        
        # chf_total
        degree = 2  # Change this to try different degrees
        coefs = Polynomial.fit(installation_cost_df['kw'], installation_cost_df['chf_total'], deg=degree).convert().coef
        def func_chf_total_poly(x, coefs):
            return sum(c * x**i for i, c in enumerate(coefs))
        estim_instcost_chftotal = lambda x: func_chf_total_poly(x, coefs)
        print_to_logfile(f'coefs: {coefs}', log_file_name_def)
        

        # plot installation cost df + intrapolation functions -------------------
        if True: 
            fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
            kw_range = np.linspace(installation_cost_df['kw'].min(), installation_cost_df['kw'].max(), 100)
            chf_pkW_fitted = estim_instcost_chfpkW(kw_range)
            chf_total_fitted = estim_instcost_chftotal(kw_range)

            # Scatter plots + interpolation
            axs[0].plot(kw_range, chf_pkW_fitted, label='Interpolated chf_pkW', color='red')  # Interpolated line
            axs[0].scatter(installation_cost_df['kw'], installation_cost_df['chf_pkW'], label='chf_pkW', color='blue')
            axs[0].set(xlabel='kW', ylabel='CHF', title='Cost per kW')
            axs[0].legend()

            axs[1].plot(kw_range, chf_total_fitted, label='Interpolated chf_total', color='green')  # Interpolated line
            axs[1].scatter(installation_cost_df['kw'], installation_cost_df['chf_total'], label='chf_total', color='orange')
            axs[1].set(xlabel='kW', ylabel='CHF', title='Total Cost')
            axs[1].legend()

            # Export the plots
            plt.tight_layout()
            plt.show()
            plt.savefig(f'{data_path_def}/output/preprep_data/pvinstcost_table.png')

        # export cost df -------------------
        installation_cost_df.to_parquet(f'{data_path_def}/output/preprep_data/pvinstcost_table.parquet')
        installation_cost_df.to_csv(f'{data_path_def}/output/preprep_data/pvinstcost_table.csv')
        checkpoint_to_logfile(f'exported pvinstcost_table', log_file_name_def=log_file_name_def, n_tabs_def = 5)


    # ATTACH CUMULATIVE COST TO ROOF PARTITIONS ================================================================

    # import and prepare solkat data -------------------
    solkat = pd.read_parquet(f'{data_path_def}/output/preprep_data/solkat.parquet')

    # transform IDs cols to str 
    def convert_srs_to_str(df, colname):
        df[colname] = df[colname].fillna(-1).astype(int).astype(str)
        df[colname] = df[colname].replace('-1', np.nan)
        return df
    solkat = convert_srs_to_str(solkat, 'EGID')
    solkat = convert_srs_to_str(solkat, 'BFS_NUMMER')
    solkat['n_partition'] = 1  # set to 1 for each individual partition, used to count partitions in groupby later
    checkpoint_to_logfile(f'imported + transformed solkat for cost extension', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)
    
    if smaller_import_def:
        solkat = solkat[0:200]


    # extend COST for ALL partition BY EGID -------------------
    solkat_egid_total = solkat.groupby(['EGID', ]).agg(
        {'FLAECHE': 'sum', 'MSTRAHLUNG': 'mean', 'GSTRAHLUNG': 'sum', 'STROMERTRAG': 'sum', 'n_partition':'sum'}).reset_index()
    solkat_egid_total['pvpot_bysurface_kw'] = solkat_egid_total['FLAECHE'] * conversion_m2_to_kw
    solkat_egid_total['cost_chf_pkW_times_kw'] = estim_instcost_chfpkW(solkat_egid_total['pvpot_bysurface_kw']) * solkat_egid_total['pvpot_bysurface_kw']
    solkat_egid_total['cost_chf_total'] = estim_instcost_chftotal(solkat_egid_total['pvpot_bysurface_kw'])

    solkat_egid_total.to_parquet(f'{data_path_def}/output/preprep_data/solkat_egid_total.parquet')
    checkpoint_to_logfile(f'exported solkat_egid_total', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def= show_debug_prints_def)


    # extend COST per ADDITIONAL partition -------------------
    # THIS COMPUTATION DOES NOT YIELD MUCH INFORMATION BECAUSE IT STILL DOES NOT GIVE THE COST FOR EACH UNIQUE PARTITION COMBINATION
    # >> Rather export the cost estimation functions so they can be called later when all the unique combinations are created. 
    if False:  
        # prepare df; sort and add counter
        solkat_egid_cumsum = solkat.loc[solkat['EGID'].notna(), ['EGID', 'GSTRAHLUNG', 'FLAECHE', 'MSTRAHLUNG', 'STROMERTRAG']].copy()
        solkat_egid_cumsum = solkat.sort_values(by=['EGID', 'GSTRAHLUNG'], ascending=[True, False]) 
        solkat_egid_cumsum['partition_counter'] = solkat_egid_cumsum.groupby('EGID').cumcount() + 1

        # apply cummulative sum in groupby on copy of variable of interest
        cumulative_cols = ['FLAECH_cumm', 'GSTRAH_cumm', 'STROME_cumm']
        solkat_egid_cumsum[cumulative_cols] = solkat_egid_cumsum[['FLAECHE', 'GSTRAHLUNG', 'STROMERTRAG']]
        for col in cumulative_cols:
            checkpoint_to_logfile(f'start cumulative sum for: {col}', log_file_name_def=log_file_name_def, n_tabs_def=2, show_debug_prints_def=show_debug_prints_def)
            solkat_egid_cumsum[col] = solkat_egid_cumsum.groupby('EGID')[col].transform(pd.Series.cumsum)

        solkat_egid_cumsum.to_parquet(f'{data_path_def}/output/preprep_data/solkat_egid_cumsum.parquet')
        checkpoint_to_logfile(f'exported solkat_egid_cumsum', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def=show_debug_prints_def)


