# -----------------------------------------------------------------------------
# MASTER_data_aggregation.py
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: This file is the master file for data aggregation. It calls all
#   necessary functions to get data from API sources and aggregate it together with
#   other locally stored spatial data sources. It does so by converting all data to 
#   parquet files (faster imports) and creating mappings for fast lookups (no single data base). 


# PACKAGES --------------------------------------------------------------------
if True:
    import os as os
    import sys
    sys.path.append(os.getcwd())

    # external packages
    import pandas as pd
    import geopandas as gpd
    import glob
    import shutil
    import winsound
    import subprocess
    from datetime import datetime
    from pprint import pprint, pformat

    # own packages and functions
    import auxiliary_functions
    from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr, format_MASTER_settings
    import data_aggregation.default_settings as dataagg_default_sett

    import data_aggregation.split_data_geometry as split_data_geometry
    import data_aggregation.api_pvtarif as api_pvtarif
    import data_aggregation.get_elecpri_data as get_elecpri_data
    import data_aggregation.sql_gwr as sql_gwr
    import data_aggregation.extend_data as extend_data

    import data_aggregation.preprepare_data as preprep_data


def MASTER_data_aggregation(dataagg_settings_func):
    """
    Input: 
        (raw input data locally stored)
        dict: dataagg_settings_func
            > settings for data aggregation scenarios

    Output (no function return but export to dir):
        > directory renamed after scenario name (dataag_scneario), containing all preprepared data from raw local input files and APIs exports. 
    
    Description:
        > Depending on settings, the all geo data sources are split into geometries and remaining data content to allow for much faster and efficent file types. 
        > Next, the script accesses all data sources (covering all of Switzerland in the recent 2015 to 2023s, with a few exceptions) and extracts all observations within 
          the specified sample range (BFS municipalities) e.g. PV feedin tariffs by DSO, electricity prices, all buildings (GebaeudeWohnungregister) and roof partitions 
          (Sonnendach, surface, angle, tilt, yearly production, monthly radiation transformation ).
          All data sources are filtered depending on the selection criteria defined in the scenario settings. 
        > Next, the script extracts all relevant time series data (NETFLEX, synthetic load profiles for a given year) and radiation data for Basel BS (Meteoblue, hourly 
          for a given weather year 2018 - 2023). Four load archetypes are matched to all residential buildings within the selected sample (dependent on electricity 
          demand proxy of GWR)
        > All data files are exported in a seperate directory for furhter usage. 
    """    


    # SETTIGNS --------------------------------------------------------------------
    
    if not isinstance(dataagg_settings_func, dict):
        print('  USE LOCAL DATA AGGREGATION SETTINGS - DICT')
        dataagg_settings = dataagg_default_sett.get_default_dataag_settings()
    else:
        dataagg_settings = dataagg_settings_func


    # SETUP -----------------------------------------------------------------------
    # set working directory
    if True:
        # wd_path = dataagg_settings['wd_path_laptop'] if not dataagg_settings['script_run_on_server'] else dataagg_settings['wd_path_server']
        wd_path = os.getcwd()
        data_path = f'{wd_path}_data'
            
        # create directory + log file
        preprep_path = f'{data_path}/preprep_data/preprep_data__temp_to_be_renamed' 
        if not os.path.exists(preprep_path):
            os.makedirs(preprep_path)
        log_name = f'{preprep_path}/preprep_data_log.txt'
        total_runtime_start = datetime.now()

        summary_name = f'{preprep_path}/summary_data_selection_log.txt'

        chapter_to_logfile(f'OptimalPV - Sample Summary of Building Topology', summary_name, overwrite_file=True)
        subchapter_to_logfile(f'MASTER_data_aggregation', summary_name)



        # get bfs numbers from canton selection if applicable
        if not not dataagg_settings['kt_numbers']: 
            dataagg_settings['bfs_numbers'] = auxiliary_functions.get_bfs_from_ktnr(dataagg_settings['kt_numbers'], data_path, log_name)
            print_to_logfile(f' > no. of kt  numbers in selection: {len(dataagg_settings["kt_numbers"])}', log_name)
            print_to_logfile(f' > no. of bfs numbers in selection: {len(dataagg_settings["bfs_numbers"])}', log_name) 

        # add information to dataagg_settings that's relevant for further functions
        dataagg_settings['log_file_name'] = log_name
        dataagg_settings['summary_file_name'] = summary_name
        dataagg_settings['wd_path'] = wd_path
        dataagg_settings['data_path'] = data_path
        dataagg_settings['preprep_path'] = preprep_path

    chapter_to_logfile(f'start MASTER_data_aggregation', log_name, overwrite_file=True)
    formated_dataagg_settings = format_MASTER_settings(dataagg_settings)
    print_to_logfile(f' > settings: \n{pformat(formated_dataagg_settings)}', log_name)



    # SPLIT DATA AND GEOMETRY ------------------------------------------------------
    # split data and geometry to avoid memory issues when importing data
    if dataagg_settings['split_data_geometry_AND_slow_api']:
        subchapter_to_logfile('pre-prep data: SPLIT DATA GEOMETRY + IMPORT SLOW APIs', log_name)
        split_data_geometry.split_data_and_geometry(dataagg_settings_def = dataagg_settings)
        split_data_geometry.get_kt_bsblso_sql_gwr(dataagg_settings_def = dataagg_settings)


    # GET SLOW API DATA ---------------------------------------------------------------
    if dataagg_settings['split_data_geometry_AND_slow_api']:
        subchapter_to_logfile('pre-prep data: API GM by EWR MAPPING', log_name)
        api_pvtarif.api_pvtarif_gm_ewr_Mapping(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: API PVTARIF', log_name)
        api_pvtarif.api_pvtarif_data(dataagg_settings_def=dataagg_settings)

        subchapter_to_logfile('pre-prep data: API ENTSOE DayAhead', log_name)
        # api_entsoe_ahead_elecpri_data(dataagg_settings_def = dataagg_settings)


    # IMPORT API DATA ---------------------------------------------------------------
    subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
    get_elecpri_data.get_elecpri_data_earlier_api_import(dataagg_settings_def = dataagg_settings)

    subchapter_to_logfile('pre-prep data: API INPUT DATA', log_name)
    preprep_data.get_earlier_api_import_data(dataagg_settings_def = dataagg_settings)


    # IMPORT LOCAL DATA + SPATIAL MAPPINGS ------------------------------------------
    # transform spatial data to parquet files for faster import and transformation
    pq_dir_exists_TF = os.path.exists(f'{preprep_path}')
    pq_files_rerun = dataagg_settings['rerun_localimport_and_mappings']

    if not pq_dir_exists_TF or pq_files_rerun:
        subchapter_to_logfile('pre-prep data: SQL GWR DATA', log_name)
        # sql_gwr.sql_gwr_data(dataagg_settings_def=dataagg_settings)

        subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', log_name)
        # local_data_to_parquet_AND_create_spatial_mappings_bySBUUID(dataagg_settings_def = dataagg_settings)
        preprep_data.local_data_AND_spatial_mappings(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: IMPORT DEMAND TS + match series HOUSES', log_name)
        preprep_data.import_demand_TS_AND_match_households(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: IMPORT METEO SUNSHINE TS', log_name)
        preprep_data.import_meteo_data(dataagg_settings_def = dataagg_settings)


    else: 
        print_to_logfile('\n', log_name)
        checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)



    # EXTEND WITH TIME FIXED DATA ---------------------------------------------------------------
    reextend_fixed_data = dataagg_settings['reextend_fixed_data']

    if reextend_fixed_data: # or not cost_df_exists_TF:
        subchapter_to_logfile('extend data: ESTIM PV INSTALLTION COST FUNCTION,  ', log_name)
        extend_data.estimate_pv_cost(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('extend data: GET ANGLE+TILT FACTOR + NODE MAPPING', log_name)
        extend_data.get_angle_tilt_table(dataagg_settings_def = dataagg_settings)
        
    
    # -----------------------------------------------------------------------------
    # END 
    chapter_to_logfile(f'END MASTER_data_aggregation\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)

    if not dataagg_settings['script_run_on_server']:
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 1000)

    
    # COPY & RENAME AGGREGATED DATA FOLDER ---------------------------------------------------------------
    # > not to overwrite completed preprep folder while debugging 
    dir_dataagg_moveto = f'{data_path}/preprep_data/{dataagg_settings["name_dir_export"]}'
    if os.path.exists(dir_dataagg_moveto):
        n_same_names = len(glob.glob(f'{dir_dataagg_moveto}*'))
        old_dir_rename = f'{dir_dataagg_moveto} ({n_same_names})'
        os.rename(dir_dataagg_moveto, old_dir_rename)

    # rename log files
    name_dir_export = dataagg_settings['name_dir_export']
    os.rename(log_name, f'{preprep_path}/preprep_data_log_{name_dir_export}.txt')
    os.rename(summary_name, f'{preprep_path}/summary_data_selection_log_{name_dir_export}.txt')


    # rename preprep folder
    os.rename(preprep_path, dir_dataagg_moveto)
    


