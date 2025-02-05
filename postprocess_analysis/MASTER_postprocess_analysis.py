
# -----------------------------------------------------------------------------
# MASTER_postprocess_analysis.py
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: This file is the master file for post processing and any form of analyis 
#   that does not involve visualizations. 


# PACKAGES --------------------------------------------------------------------
if True :
    import os as os
    import sys
    sys.path.append(os.getcwd())

    # external packages
    import pandas as pd
    import geopandas as gpd
    import glob
    import shutil
    import winsound
    import json
    from datetime import datetime
    from pprint import pprint, pformat

    # own packages and functions
    from auxiliary_functions import *
    import pv_allocation.default_settings as pvalloc_default_sett
    import postprocess_analysis.default_settings as postprocess_default_sett

    import postprocess_analysis.pvalloc_to_today_sanitcheck as pvalloc_to_today_sanitcheck



def MASTER_postprocess_analysis(pvalloc_scenarios_func, postprocess_analysis_settings_func):

    # SETTINGS --------------------------------------------------------------------
    if True: 
        if not isinstance(pvalloc_scenarios_func, dict):
            print(' USE LOCAL SETTINGS - DICT  ')
            pvalloc_scenarios = pvalloc_default_sett.get_default_pvalloc_settings()
        else:
            pvalloc_scenarios = pvalloc_scenarios_func
            
        if not isinstance(postprocess_analysis_settings_func, dict) or postprocess_analysis_settings_func == {}:
            postprocess_analysis_settings = postprocess_default_sett.get_default_postprocess_analysis_settings()
        else:
            postprocess_analysis_settings = postprocess_analysis_settings_func
        
        pvalloc_sett_run_on_server = pvalloc_scenarios.get(next(iter(pvalloc_scenarios))).get('script_run_on_server')

        
    # SETUP -----------------------------------------------------------------------
    if True: 
        # general setup for paths etc.
        # first_alloc_sett = pvalloc_scenarios[list(pvalloc_scenarios.keys())[0]]
        # wd_path = first_alloc_sett['wd_path_laptop'] if not first_alloc_sett['script_run_on_server'] else first_alloc_sett['wd_path_server']
        wd_path = os.getcwd()
        data_path = f'{wd_path}_data'

        # create directory + log file
        postprocess_path = f'{data_path}/output/postprocess_analysis'
        if not os.path.exists(postprocess_path):
            os.makedirs(postprocess_path)

        log_name = f'{data_path}/output/postprocess_analysis_log.txt'
        total_runtime_start = datetime.now()


        # extract scenario settings + information ------------------------
        scen_dir_export_list, pvalloc_scen_list = [], []
        # scen_dir_import_list, T0_prediction_list, months_prediction_list = [], [], [], [] T0_prediction_list, months_lookback_list, months_prediction_list = [], [], [] pvalloc_scen_list = []
        for key, val in pvalloc_scenarios.items():
            pvalloc_settings_path = glob.glob(f'{data_path}/output/{key}/pvalloc_settings.json')
            
            if len(pvalloc_settings_path) == 1:
                try:
                    scen_sett = json.load(open(pvalloc_settings_path[0], 'r'))
                    pvalloc_scen_list.append(scen_sett)
                    scen_dir_export_list.append(scen_sett['name_dir_export'])
                except:
                    print(f'ERROR: could not load pvalloc_settings.json for {key}, take function input')
                    pvalloc_scen_list.append(val)
                    scen_dir_export_list.append(val['name_dir_export'])

            else:
                pvalloc_scen_list.append(val)
                scen_dir_export_list.append(val['name_dir_export'])
    

    chapter_to_logfile(f'start MASTER_postprocess_analysis', log_name, overwrite_file=True)




    # POSTPROCESSING ==============================================================
    # ...



    # ANALYSIS ====================================================================
    pvalloc_to_today_sanitcheck.prediction_accuracy(pvalloc_scen_list, postprocess_analysis_settings, wd_path, data_path, log_name)




    # END  ================================================================
    chapter_to_logfile(f'END MASTER_postprocess_analysis\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)
    if not pvalloc_sett_run_on_server:
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 1000)