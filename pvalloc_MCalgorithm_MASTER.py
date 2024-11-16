# -----------------------------------------------------------------------------
# pvalloc_MCalgorithm_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 


# PACKAGES --------------------------------------------------------------------
if True:
    import os as os
    import sys
    sys.path.append(os.getcwd())
    # sys.path.append(pvalloc_settings['wd_path_laptop']) if pvalloc_settings['script_run_on_server'] else sys.path.append(pvalloc_settings['wd_path_server'])

    # external packages
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from datetime import datetime
    from pprint import pformat

    import glob
    import shutil
    import winsound

    # own packages and functions
    import auxiliary_functions
    from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr, format_MASTER_settings
    import pv_allocation.default_settings as pvalloc_default_sett


def pvalloc_MC_algorithm_MASTER(pvalloc_settings_func):

    # SETTIGNS --------------------------------------------------------------------
    if not isinstance(pvalloc_settings_func, dict):
        print(' USE LOCAL SETTINGS - DICT  ')
        pvalloc_settings =  pvalloc_default_sett.get_default_pvalloc_settings()
    else:
        pvalloc_settings = pvalloc_settings_func


    # SETUP ================================================================
    if True: 
        # set working directory
        wd_path = pvalloc_settings['wd_path_laptop'] if not pvalloc_settings['script_run_on_server'] else pvalloc_settings['wd_path_server']
        data_path = f'{wd_path}_data'

        # create directory + log file
        pvalloc_path = f'{data_path}/output/pvalloc_run'
        if not os.path.exists(pvalloc_path):
            os.makedirs(pvalloc_path)
        log_name = f'{data_path}/output/pvalloc_MCalgo_log.txt'
        total_runtime_start = datetime.now()

    
    # MONTE CARLO ITERATION LOOP ================================================================================
    
    # CREATE MC DIR + TRANSFER INITIAL DATA FILES ------------------------------------------
    # pvalloc_settings["name_dir_export"] = 'pvalloc_DEV_12m_pvmethod2_selfconsum00'
    montecarlo_iterations = pvalloc_settings['MC_loop_specs']['montecarlo_iterations']
    fresh_initial_files = pvalloc_settings['MC_loop_specs']['fresh_initial_files']
    
    montecarlo_iterations = [1,3,11,33,]
    fresh_initial_files = ['topo_egid.json', 'months_prediction.parquet', 'gridprem_ts.parquet', 'gridnode_df.parquet', 'constrcapa.parquet']

    # get all initial files to start a fresh MC iteration
    fresh_initial_paths = [f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/{file}' for file in fresh_initial_files]
    topo_time_paths = glob.glob(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/topo_time_subdf/*.parquet')
    all_initial_paths = fresh_initial_paths + topo_time_paths
    
    # create MC directories
    if not os.path.exists(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/MC_iterations'):
        os.makedirs(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/MC_iterations')

    max_digits = len(str(max(montecarlo_iterations)))
    mc_iter = montecarlo_iterations[0]
    # for mc_iter in montecarlo_iterations:
    if True:
        # mc_data_path = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/MC_iterations/MC_{mc_iter:0{max_digits}}'
        mc_data_path = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/zMC_{mc_iter:0{max_digits}}'
        os.makedirs(mc_data_path, exist_ok=True)

        # copy all initial files to MC directory
        for file in all_initial_paths:
            shutil.copy(file, mc_data_path)

        # remove old files to avoid concatenating old files to iteration-by-iteration interim saves
        for df_type in ['npv_df', 'pred_inst_df']:
            df_paths = glob.glob(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/{df_type}.*')
            for f in df_paths:
                os.remove(f)
          
    # ALLOCATION ALGORITHM ----------------------------------------------                 
    # empty lists and dfs for aggregation later
    dfuid_installed_list = []
    pred_inst_df = pd.DataFrame()   

    # import all required files
    # topo_egid = json.load(open(f'{mc_data_path}/topo_egid.json', 'r'))

    safety_counter_max = pvalloc_settings['algorithm_specs']['while_inst_counter_max']

    glob.glob(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/topo_time_subdf/*.parquet')

    # BOOKMARK!
