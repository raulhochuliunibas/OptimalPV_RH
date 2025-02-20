import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from collections import OrderedDict
from numpy.polynomial.polynomial import Polynomial


sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# INVESTMENT COSTS
# ------------------------------------------------------------------------------------------------------

def estimate_pv_cost(
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
    preprep_path_def = dataagg_settings_def['preprep_path']


    print_to_logfile(f'run function: attach_pv_cost.py', log_file_name_def= log_file_name_def)


    # CREATE COST DF AND FUNCTIONS ================================================================
    if True:
        # conversion_m2_to_kw = 0.1  # A 1m2 area can fit 0.1 kWp of PV Panels
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
        checkpoint_to_logfile(f'created intrapolation function for chf_pkW using "cureve_fit" to receive curve parameters', log_file_name_def)
        print_to_logfile(f'params_pkW: {params_pkW}', log_file_name_def)
        
        # chf_total
        degree = 2  # Change this to try different degrees
        coefs_total = Polynomial.fit(installation_cost_df['kw'], installation_cost_df['chf_total'], deg=degree).convert().coef
        def func_chf_total_poly(x, coefs_total):
            return sum(c * x**i for i, c in enumerate(coefs_total))
        estim_instcost_chftotal = lambda x: func_chf_total_poly(x, coefs_total)
        checkpoint_to_logfile(f'created intrapolation function for chf_total using "Polynomial.fit" to receive curve coefficients', log_file_name_def)
        print_to_logfile(f'coefs_total: {coefs_total}', log_file_name_def)

        pvinstcost_coefficients = {
            'params_pkW': list(params_pkW),
            'coefs_total': list(coefs_total)
        }

        # export 
        with open(f'{preprep_path_def}/pvinstcost_coefficients.json', 'w') as f:
            json.dump(pvinstcost_coefficients, f)

        np.save(f'{preprep_path_def}/pvinstcost_coefficients.npy', pvinstcost_coefficients)


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
            # plt.show()
            plt.savefig(f'{preprep_path_def}/pvinstcost_table.png')

        # export cost df -------------------
        installation_cost_df.to_parquet(f'{preprep_path_def}/pvinstcost_table.parquet')
        installation_cost_df.to_csv(f'{preprep_path_def}/pvinstcost_table.csv')
        checkpoint_to_logfile(f'exported pvinstcost_table', log_file_name_def=log_file_name_def, n_tabs_def = 5)


    # ATTACH CUMULATIVE COST TO ROOF PARTITIONS ================================================================
    # IT MAKES no sense to calculate the cost for EACH partition! Most partitions will not be considered any way 
    # and the cost need to be intrapolated for each combination as well, chaging the cost for each partition because
    # costs are assumed to be non-linear!

    if False: 
        # import and prepare solkat data -------------------
        solkat = pd.read_parquet(f'{preprep_path_def}/solkat.parquet')

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

        solkat_egid_total.to_parquet(f'{preprep_path_def}/solkat_egid_total.parquet')
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

        solkat_egid_cumsum.to_parquet(f'{preprep_path_def}/solkat_egid_cumsum.parquet')
        checkpoint_to_logfile(f'exported solkat_egid_cumsum', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def=show_debug_prints_def)



# ------------------------------------------------------------------------------------------------------
# ANGLE TILT & AZIMUTH Table
# ------------------------------------------------------------------------------------------------------
def get_angle_tilt_table(dataagg_settings_def):

    # import settings + setup -------------------
    data_path_def = dataagg_settings_def['data_path']
    preprep_path_def = dataagg_settings_def['preprep_path']
    log_file_name_def = dataagg_settings_def['log_file_name']
    # name_dir_import_def = pvalloc_settings['name_dir_import']
    print_to_logfile('run function: get_angle_tilt_table', log_file_name_def)

    # SOURCE: table was retreived from this site: https://echtsolar.de/photovoltaik-neigungswinkel/
    # date 29.08.24
    
    # import df ---------
    index_angle = [-180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    index_tilt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    tuples_iter = list(itertools.product(index_angle, index_tilt))

    tuples = [(-180, 0), (-180, 5), (-180, 10), (-180, 15), (-180, 20), (-180, 25), (-180, 30), (-180, 35), (-180, 40), (-180, 45), (-180, 50), (-180, 55), (-180, 60), (-180, 65), (-180, 70), (-180, 75), (-180, 80), (-180, 85), (-180, 90), 
              (-170, 0), (-170, 5), (-170, 10), (-170, 15), (-170, 20), (-170, 25), (-170, 30), (-170, 35), (-170, 40), (-170, 45), (-170, 50), (-170, 55), (-170, 60), (-170, 65), (-170, 70), (-170, 75), (-170, 80), (-170, 85), (-170, 90), 
              (-160, 0), (-160, 5), (-160, 10), (-160, 15), (-160, 20), (-160, 25), (-160, 30), (-160, 35), (-160, 40), (-160, 45), (-160, 50), (-160, 55), (-160, 60), (-160, 65), (-160, 70), (-160, 75), (-160, 80), (-160, 85), (-160, 90), 
              (-150, 0), (-150, 5), (-150, 10), (-150, 15), (-150, 20), (-150, 25), (-150, 30), (-150, 35), (-150, 40), (-150, 45), (-150, 50), (-150, 55), (-150, 60), (-150, 65), (-150, 70), (-150, 75), (-150, 80), (-150, 85), (-150, 90), 
              (-140, 0), (-140, 5), (-140, 10), (-140, 15), (-140, 20), (-140, 25), (-140, 30), (-140, 35), (-140, 40), (-140, 45), (-140, 50), (-140, 55), (-140, 60), (-140, 65), (-140, 70), (-140, 75), (-140, 80), (-140, 85), (-140, 90),
              (-130, 0), (-130, 5), (-130, 10), (-130, 15), (-130, 20), (-130, 25), (-130, 30), (-130, 35), (-130, 40), (-130, 45), (-130, 50), (-130, 55), (-130, 60), (-130, 65), (-130, 70), (-130, 75), (-130, 80), (-130, 85), (-130, 90),
              (-120, 0), (-120, 5), (-120, 10), (-120, 15), (-120, 20), (-120, 25), (-120, 30), (-120, 35), (-120, 40), (-120, 45), (-120, 50), (-120, 55), (-120, 60), (-120, 65), (-120, 70), (-120, 75), (-120, 80), (-120, 85), (-120, 90),
              (-110, 0), (-110, 5), (-110, 10), (-110, 15), (-110, 20), (-110, 25), (-110, 30), (-110, 35), (-110, 40), (-110, 45), (-110, 50), (-110, 55), (-110, 60), (-110, 65), (-110, 70), (-110, 75), (-110, 80), (-110, 85), (-110, 90),
              (-100, 0), (-100, 5), (-100, 10), (-100, 15), (-100, 20), (-100, 25), (-100, 30), (-100, 35), (-100, 40), (-100, 45), (-100, 50), (-100, 55), (-100, 60), (-100, 65), (-100, 70), (-100, 75), (-100, 80), (-100, 85), (-100, 90),
              (-90, 0), (-90, 5), (-90, 10), (-90, 15), (-90, 20), (-90, 25), (-90, 30), (-90, 35), (-90, 40), (-90, 45), (-90, 50), (-90, 55), (-90, 60), (-90, 65), (-90, 70), (-90, 75), (-90, 80), (-90, 85), (-90, 90),
              (-80, 0), (-80, 5), (-80, 10), (-80, 15), (-80, 20), (-80, 25), (-80, 30), (-80, 35), (-80, 40), (-80, 45), (-80, 50), (-80, 55), (-80, 60), (-80, 65), (-80, 70), (-80, 75), (-80, 80), (-80, 85), (-80, 90),
              (-70, 0), (-70, 5), (-70, 10), (-70, 15), (-70, 20), (-70, 25), (-70, 30), (-70, 35), (-70, 40), (-70, 45), (-70, 50), (-70, 55), (-70, 60), (-70, 65), (-70, 70), (-70, 75), (-70, 80), (-70, 85), (-70, 90),
              (-60, 0), (-60, 5), (-60, 10), (-60, 15), (-60, 20), (-60, 25), (-60, 30), (-60, 35), (-60, 40), (-60, 45), (-60, 50), (-60, 55), (-60, 60), (-60, 65), (-60, 70), (-60, 75), (-60, 80), (-60, 85), (-60, 90),
              (-50, 0), (-50, 5), (-50, 10), (-50, 15), (-50, 20), (-50, 25), (-50, 30), (-50, 35), (-50, 40), (-50, 45), (-50, 50), (-50, 55), (-50, 60), (-50, 65), (-50, 70), (-50, 75), (-50, 80), (-50, 85), (-50, 90),
              (-40, 0), (-40, 5), (-40, 10), (-40, 15), (-40, 20), (-40, 25), (-40, 30), (-40, 35), (-40, 40), (-40, 45), (-40, 50), (-40, 55), (-40, 60), (-40, 65), (-40, 70), (-40, 75), (-40, 80), (-40, 85), (-40, 90),
              (-30, 0), (-30, 5), (-30, 10), (-30, 15), (-30, 20), (-30, 25), (-30, 30), (-30, 35), (-30, 40), (-30, 45), (-30, 50), (-30, 55), (-30, 60), (-30, 65), (-30, 70), (-30, 75), (-30, 80), (-30, 85), (-30, 90),
              (-20, 0), (-20, 5), (-20, 10), (-20, 15), (-20, 20), (-20, 25), (-20, 30), (-20, 35), (-20, 40), (-20, 45), (-20, 50), (-20, 55), (-20, 60), (-20, 65), (-20, 70), (-20, 75), (-20, 80), (-20, 85), (-20, 90),
              (-10, 0), (-10, 5), (-10, 10), (-10, 15), (-10, 20), (-10, 25), (-10, 30), (-10, 35), (-10, 40), (-10, 45), (-10, 50), (-10, 55), (-10, 60), (-10, 65), (-10, 70), (-10, 75), (-10, 80), (-10, 85), (-10, 90),
              (0, 0), (0, 5), (0, 10), (0, 15), (0, 20), (0, 25), (0, 30), (0, 35), (0, 40), (0, 45), (0, 50), (0, 55), (0, 60), (0, 65), (0, 70), (0, 75), (0, 80), (0, 85), (0, 90),
              (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55), (10, 60), (10, 65), (10, 70), (10, 75), (10, 80), (10, 85), (10, 90),
              (20, 0), (20, 5), (20, 10), (20, 15), (20, 20), (20, 25), (20, 30), (20, 35), (20, 40), (20, 45), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (20, 75), (20, 80), (20, 85), (20, 90),
              (30, 0), (30, 5), (30, 10), (30, 15), (30, 20), (30, 25), (30, 30), (30, 35), (30, 40), (30, 45), (30, 50), (30, 55), (30, 60), (30, 65), (30, 70), (30, 75), (30, 80), (30, 85), (30, 90),
              (40, 0), (40, 5), (40, 10), (40, 15), (40, 20), (40, 25), (40, 30), (40, 35), (40, 40), (40, 45), (40, 50), (40, 55), (40, 60), (40, 65), (40, 70), (40, 75), (40, 80), (40, 85), (40, 90),
              (50, 0), (50, 5), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30), (50, 35), (50, 40), (50, 45), (50, 50), (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90),
              (60, 0), (60, 5), (60, 10), (60, 15), (60, 20), (60, 25), (60, 30), (60, 35), (60, 40), (60, 45), (60, 50), (60, 55), (60, 60), (60, 65), (60, 70), (60, 75), (60, 80), (60, 85), (60, 90),
              (70, 0), (70, 5), (70, 10), (70, 15), (70, 20), (70, 25), (70, 30), (70, 35), (70, 40), (70, 45), (70, 50), (70, 55), (70, 60), (70, 65), (70, 70), (70, 75), (70, 80), (70, 85), (70, 90),
              (80, 0), (80, 5), (80, 10), (80, 15), (80, 20), (80, 25), (80, 30), (80, 35), (80, 40), (80, 45), (80, 50), (80, 55), (80, 60), (80, 65), (80, 70), (80, 75), (80, 80), (80, 85), (80, 90),
              (90, 0), (90, 5), (90, 10), (90, 15), (90, 20), (90, 25), (90, 30), (90, 35), (90, 40), (90, 45), (90, 50), (90, 55), (90, 60), (90, 65), (90, 70), (90, 75), (90, 80), (90, 85), (90, 90),
              (100, 0), (100, 5), (100, 10), (100, 15), (100, 20), (100, 25), (100, 30), (100, 35), (100, 40), (100, 45), (100, 50), (100, 55), (100, 60), (100, 65), (100, 70), (100, 75), (100, 80), (100, 85), (100, 90),
              (110, 0), (110, 5), (110, 10), (110, 15), (110, 20), (110, 25), (110, 30), (110, 35), (110, 40), (110, 45), (110, 50), (110, 55), (110, 60), (110, 65), (110, 70), (110, 75), (110, 80), (110, 85), (110, 90),
              (120, 0), (120, 5), (120, 10), (120, 15), (120, 20), (120, 25), (120, 30), (120, 35), (120, 40), (120, 45), (120, 50), (120, 55), (120, 60), (120, 65), (120, 70), (120, 75), (120, 80), (120, 85), (120, 90),
              (130, 0), (130, 5), (130, 10), (130, 15), (130, 20), (130, 25), (130, 30), (130, 35), (130, 40), (130, 45), (130, 50), (130, 55), (130, 60), (130, 65), (130, 70), (130, 75), (130, 80), (130, 85), (130, 90),
              (140, 0), (140, 5), (140, 10), (140, 15), (140, 20), (140, 25), (140, 30), (140, 35), (140, 40), (140, 45), (140, 50), (140, 55), (140, 60), (140, 65), (140, 70), (140, 75), (140, 80), (140, 85), (140, 90),
              (150, 0), (150, 5), (150, 10), (150, 15), (150, 20), (150, 25), (150, 30), (150, 35), (150, 40), (150, 45), (150, 50), (150, 55), (150, 60), (150, 65), (150, 70), (150, 75), (150, 80), (150, 85), (150, 90),
              (160, 0), (160, 5), (160, 10), (160, 15), (160, 20), (160, 25), (160, 30), (160, 35), (160, 40), (160, 45), (160, 50), (160, 55), (160, 60), (160, 65), (160, 70), (160, 75), (160, 80), (160, 85), (160, 90),
              (170, 0), (170, 5), (170, 10), (170, 15), (170, 20), (170, 25), (170, 30), (170, 35), (170, 40), (170, 45), (170, 50), (170, 55), (170, 60), (170, 65), (170, 70), (170, 75), (170, 80), (170, 85), (170, 90),
              (180, 0), (180, 5), (180, 10), (180, 15), (180, 20), (180, 25), (180, 30), (180, 35), (180, 40), (180, 45), (180, 50), (180, 55), (180, 60), (180, 65), (180, 70), (180, 75), (180, 80), (180, 85), (180, 90)
              ]
    index = pd.MultiIndex.from_tuples(tuples, names=['angle', 'tilt'])

    values = [89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2, 
              89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
              89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
              89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
              89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
              89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
              89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
              89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
              89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
              89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
              89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
              89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
              89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
              89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
              89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
              89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
              89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
              89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
              89.0, 92.1, 94.7, 96.8, 98.4, 99.5, 100,  100 , 99.5, 98.3, 96.8, 94.8, 92.3, 89.3, 85.8, 81.9, 77.6, 73.1, 68.1,
              89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
              89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
              89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
              89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
              89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
              89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
              89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
              89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
              89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
              89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
              89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
              89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
              89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
              89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
              89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
              89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
              89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
              89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2
              ] 
    
    angle_tilt_df = pd.DataFrame(data = values, index = index, columns = ['efficiency_factor'])
    angle_tilt_df['efficiency_factor'] = angle_tilt_df['efficiency_factor'] / 100

    # export df ----------
    angle_tilt_df.to_parquet(f'{preprep_path_def}/angle_tilt_df.parquet')
    angle_tilt_df.to_csv(f'{preprep_path_def}/angle_tilt_df.csv')
    # angle_tilt_df.to_parquet(f'{data_path_def}/output/{name_dir_import_def}/angle_tilt_df.parquet')
    # return angle_tilt_df



# NO LONGER USED - grid node data is copied to preprep_data and then later processed in pvalloc_initilization_MASTER.py
# ------------------------------------------------------------------------------------------------------
# FAKE TRAFO EGID MAPPING
# ------------------------------------------------------------------------------------------------------
def get_fake_gridnodes(dataagg_settings_def):
    
    # import settings + setup -------------------
    data_path_def = dataagg_settings_def['data_path']
    preprep_path_def = dataagg_settings_def['preprep_path']
    log_file_name_def = dataagg_settings_def['log_file_name']
    # name_dir_import_def = pvalloc_settings['name_dir_import']
    print_to_logfile('run function: get_fake_gridnodes', log_file_name_def)

    # create fake gridnodes ----------------------
    # gwr = pd.read_parquet(f'{data_path_def}/output/{name_dir_import_def}/gwr.parquet')
    gwr = pd.read_parquet(f'{preprep_path_def}/gwr.parquet')

    gwr_nodes = gwr[['EGID', 'GDEKT']].copy()
    gwr_nodes['EGID_int'] = gwr_nodes['EGID'].astype(int)

    gwr_nodes.sort_values(by=['GDEKT','EGID_int'], inplace=True)
    gwr_nodes['grid_node'] = pd.cut(gwr_nodes['EGID_int'], bins=4, labels=['node1', 'node2', 'node3', 'node4'])

    gwr_nodes.drop(columns=['EGID_int', 'GDEKT'], inplace=True)
    gwr_nodes.set_index('EGID', inplace=True)

    # export df ----------
    Map_egid_nodes = gwr_nodes.copy()
    Map_egid_nodes.to_parquet(f'{preprep_path_def}/Map_egid_nodes.parquet')
    Map_egid_nodes.to_csv(f'{preprep_path_def}/Map_egid_nodes.csv')
    # Map_egid_nodes.to_parquet(f'{data_path_def}/output/{name_dir_import_def}/Map_egid_nodes.parquet')
    # return gwr_nodes



