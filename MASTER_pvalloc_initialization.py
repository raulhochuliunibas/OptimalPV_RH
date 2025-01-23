# -----------------------------------------------------------------------------
# pv_allocation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: sw
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
    
    import json
    import glob
    import shutil
    import winsound

    # own packages and functions
    import auxiliary_functions
    from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr, format_MASTER_settings
    import pv_allocation.default_settings as pvalloc_default_sett

    import pv_allocation.initialization_small_functions as initial_sml
    import pv_allocation.initialization_large_functions as  initial_lrg
    import pv_allocation.alloc_algorithm as algo
    import pv_allocation.alloc_sanitychecks as sanity
    import pv_allocation.inst_selection as select

    from pv_allocation.initialization_large_functions import *
    from pv_allocation.alloc_algorithm import *
    from pv_allocation.inst_selection import *
    from pv_allocation.alloc_sanitychecks import *
    from pv_allocation.default_settings import *


def pvalloc_initialization_MASTER(pvalloc_settings_func):

    # SETTIGNS --------------------------------------------------------------------
    if not isinstance(pvalloc_settings_func, dict):
        print(' USE LOCAL SETTINGS - DICT  ')
        pvalloc_settings =  pvalloc_default_sett.get_default_pvalloc_settings()
    else:
        pvalloc_settings = pvalloc_settings_func


    # SETUP ================================================================
    if True: 
        # set working directory
        # wd_path = pvalloc_settings['wd_path_laptop'] if not pvalloc_settings['script_run_on_server'] else pvalloc_settings['wd_path_server']
        wd_path = os.getcwd()
        data_path = f'{wd_path}_data'

        # create directory + log file
        pvalloc_path = f'{data_path}/output/pvalloc_run'
        if not os.path.exists(pvalloc_path):
            os.makedirs(pvalloc_path)
        log_name = f'{data_path}/output/pvalloc_init_log.txt'
        total_runtime_start = datetime.now()

        # transfer summary file from data_aggregation
        summary_find_path = glob.glob(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/summary_data_selection_log*.txt')
        summary_name = f'{data_path}/output/summary_data_selection_log.txt'
        if len(summary_find_path) == 1:
            shutil.copy(summary_find_path[0], summary_name)
        else:
            print_to_logfile(f' **ERROR** : summary file not found or multiple files found', log_name)


        # extend settings dict with relevant informations for later functions
        if not not pvalloc_settings['kt_numbers']:
            pvalloc_settings['bfs_numbers'] = auxiliary_functions.get_bfs_from_ktnr(pvalloc_settings['kt_numbers'], data_path, log_name)

        elif (not pvalloc_settings['kt_numbers']) and (not not pvalloc_settings['bfs_numbers']):
            pvalloc_settings['bfs_numbers'] = [str(bfs) for bfs in pvalloc_settings['bfs_numbers']]

        pvalloc_settings['log_file_name'] = log_name
        pvalloc_settings['summary_file_name'] = summary_name
        pvalloc_settings['wd_path'] = wd_path
        pvalloc_settings['data_path'] = data_path
        pvalloc_settings['pvalloc_path'] = pvalloc_path
        pvalloc_settings['interim_path'] = initial_sml.get_interim_path(pvalloc_settings)
        show_debug_prints = pvalloc_settings['show_debug_prints']


        
    # INITIALIZATION ================================================================
    chapter_to_logfile(f'start pvalloc_initialization_MASTER for: {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
    formated_pvalloc_settings = format_MASTER_settings(pvalloc_settings)
    print_to_logfile(f' > no. of kt  numbers in selection: {len(pvalloc_settings["kt_numbers"])}', log_name)
    print_to_logfile(f' > no. of bfs numbers in selection: {len(pvalloc_settings["bfs_numbers"])}', log_name) 
    print_to_logfile(f'pvalloc_settings: \n{pformat(formated_pvalloc_settings)}', log_name)
    
    # intitial print to summary file
    subchapter_to_logfile(f'pvalloc_initialization_MASTER', summary_name)

    # store settings to output folder
    with open(f'{data_path}/output/pvalloc_run/pvalloc_settings.json', 'w') as f:
        json.dump(pvalloc_settings, f, indent=4)
    with open(f'{data_path}/output/pvalloc_run/pvalloc_settings__initMASTERpy__{pvalloc_settings["name_dir_export"]}.json', 'w') as f:
        json.dump(pvalloc_settings, f, indent=4)


    
    # CREATE TOPOLOGY ================================================================
    subchapter_to_logfile('initialization: CREATE SMALLER AID DFs', log_name)
    initial_sml.HOY_weatheryear_df(pvalloc_settings)
    initial_sml.get_gridnodes_DSO(pvalloc_settings)
    
    if pvalloc_settings['recreate_topology']:
        subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', log_name)
        topo, df_list, df_names = initial.import_prepre_AND_create_topology(pvalloc_settings)

        # elif not pvalloc_settings['recreate_topology']:
        #     subchapter_to_logfile('initialization: IMPORT EXISITNG TOPOLOGY', log_name) 
        #     df_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_egid', 'Map_egid_demandtypes', 'pv', 'pvtarif', 'elecpri', 'angle_tilt_df', 'Map_egid_nodes']
        #     topo, df_list, df_names = initial.import_exisitng_topology(pvalloc_settings, df_search_names= df_names)

        subchapter_to_logfile('initialization: IMPORT TS DATA', log_name)
        ts_list, ts_names = initial.import_ts_data(pvalloc_settings)

        subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', log_name)
        constrcapa, months_prediction, months_lookback = initial.define_construction_capacity(pvalloc_settings, topo, df_list, df_names, ts_list, ts_names)



    # PREPARE TOPO_TIME SPECIFIC DFs ================================================================

    # CALC ECONOMICS for TOPO_DF ----------------------------------------------------------------
    if pvalloc_settings['recalc_economics_topo_df']:
        subchapter_to_logfile('prep: CALC ECONOMICS for TOPO_DF', log_name)
        algo.calc_economics_in_topo_df(pvalloc_settings, topo, 
                                        df_list, df_names, ts_list, ts_names)
    
        shutil.copy(f'{data_path}/output/pvalloc_run/topo_egid.json', f'{data_path}/output/pvalloc_run/topo_egid_before_alloc.json')



    # TOPOLOGY SANITY CHECKS ================================================================
    if pvalloc_settings['sanitycheck_byEGID']:
        subchapter_to_logfile('sanity_check: RUN FEW ITERATION for byCHECK', log_name)
        sanitycheck_path = f'{data_path}/output/pvalloc_run/sanity_check_byEGID'
        # make sanitycheck folder and move relevant initial files there (delete all old files, not distort results)
        if not os.path.exists(sanitycheck_path):
            os.makedirs(sanitycheck_path)
        elif os.path.exists(sanitycheck_path):
            for f in glob.glob(f'{sanitycheck_path}/*'):
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)

        fresh_initial_files = [f'{data_path}/output/pvalloc_run/{file}' for file in ['topo_egid.json', 'gridprem_ts.parquet', 'dsonodes_df.parquet']]
        topo_time_paths = glob.glob(f'{data_path}/output/pvalloc_run/topo_time_subdf/*.parquet')
        all_initial_paths = fresh_initial_files + topo_time_paths
        for f in all_initial_paths:
            shutil.copy(f, f'{sanitycheck_path}/')

        # sanity check: CALC FEW ITERATION OF NPV AND FEEDIN for check ---------------------------------------------------------------
        dfuid_installed_list = []
        pred_inst_df = pd.DataFrame()
        months_prediction_pq = pd.read_parquet(f'{data_path}/output/pvalloc_run/months_prediction.parquet')['date']
        months_prediction = [str(m) for m in months_prediction_pq]
        # i_m, m = 1, months_prediction[0:2]
        for i_m, m in enumerate(months_prediction[0:pvalloc_settings['sanitycheck_summary_byEGID_specs']['n_iterations_before_sanitycheck']]):
            print_to_logfile(f'\n-- month {m} -- sanity check -- {pvalloc_settings["name_dir_export"]} --', log_name)
            algo.update_gridprem(pvalloc_settings, sanitycheck_path, m, i_m)
            algo.update_npv_df(pvalloc_settings, sanitycheck_path, m, i_m)
            select.select_AND_adjust_topology(pvalloc_settings, sanitycheck_path,
                                            dfuid_installed_list,pred_inst_df,
                                            m, i_m)
        
        sanity.sanity_check_summary_byEGID(pvalloc_settings, sanitycheck_path)


    # sanity check: CREATE MAP OF TOPO_DF ----------------------------------------------------------------
    if pvalloc_settings['create_gdf_export_of_topology']:
        subchapter_to_logfile('sanity_check: CREATE SPATIAL EXPORTS OF TOPOLOGY_DF', log_name)
        sanity.create_gdf_export_of_topology(pvalloc_settings)  


    # sanity check: CREATE MAP OF TOPO_DF ----------------------------------------------------------------
    subchapter_to_logfile('sanity_check: MULTIPLE INSTALLATIONS PER EGID', log_name)
    sanity.check_multiple_xtf_ids_per_EGID(pvalloc_settings)


    # END  ================================================================
    chapter_to_logfile(f'END pvalloc_initialization_MASTER\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)
    if not pvalloc_settings['script_run_on_server']:
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 1000)

    # COPY & RENAME PVALLOC DATA FOLDER ---------------------------------------------------------------
    # > not to overwrite completed folder while debugging 
    dir_alloc_moveto = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}'
    if os.path.exists(dir_alloc_moveto):
        n_same_names = len(glob.glob(f'{dir_alloc_moveto}*'))
        old_dir_rename = f'{dir_alloc_moveto} ({n_same_names+1})'
        os.rename(f'{dir_alloc_moveto}', old_dir_rename)

    # os.makedirs(dir_alloc_moveto)
    # file_to_move = glob.glob(f'{data_path}/output/pvalloc_run/*')
    # for f in file_to_move:
    #     if os.path.isfile(f):
    #         shutil.copy(f, dir_alloc_moveto)
    #     elif os.path.isdir(f):
    #         shutil.copytree(f, os.path.join(dir_alloc_moveto, os.path.basename(f)))
    os.rename(f'{data_path}/output/pvalloc_run', dir_alloc_moveto)
    # shutil.copy(glob.glob(f'{data_path}/output/pvalloc_init_log.txt')[0], f'{dir_alloc_moveto}/pvalloc_init_log_{pvalloc_settings["name_dir_export"]}.txt')
    shutil.move(glob.glob(f'{data_path}/output/pvalloc_init_log.txt')[0], f'{dir_alloc_moveto}/pvalloc_init_log_{pvalloc_settings["name_dir_export"]}.txt')









