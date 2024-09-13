# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
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

    from data_aggregation.split_data_geometry import *
    from data_aggregation.api_electricity_prices import *
    from data_aggregation.sql_gwr import *
    from data_aggregation.api_pvtarif import *
    from data_aggregation.api_entsoe import *
    from data_aggregation.preprepare_data import local_data_AND_spatial_mappings, local_data_to_parquet_AND_create_spatial_mappings_bySBUUID, import_demand_TS_AND_match_households, import_meteo_data
    from data_aggregation.extend_data import *

dataagg_settings_local = {
    'name_dir_export': 'preprep_BSBLSO_18to22',     # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
    'script_run_on_server': False,                  # F: run on private computer, T: run on server
    'smaller_import': False,                	        # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
    'show_debug_prints': True,                      # F: certain print statements are omitted, T: includes print statements that help with debugging
    'turnoff_comp_after_run': False,                # F: keep computer running after script is finished, T: turn off computer after script is finished
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',     # path to the working directory on Raul's laptop
    'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH', # path to the working directory on the server

    'kt_numbers': [11,12,13],                       # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
    'bfs_numbers': [],                              # list of municipalites to select for allocation (only used if kt_numbers == 0)
    'year_range': [2018, 2022],                     # range of years to import
    
    # switch on/off parts of aggregation
    'split_data_geometry_AND_slow_api': True, 
    # 'reimport_api_data_1': True,                   # F: use existing parquet files, T: recreate parquet files in data prep        
    'reimport_api_data': True,
    'rerun_localimport_and_mappings': True,         # F: use existi ng parquet files, T: recreate parquet files in data prep
    'reextend_fixed_data': True,                    # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again       
    
    # settings for gwr selection
    'gwr_selection_specs': {
        'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                    'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                    'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
        'dwelling_cols':['EGID', 'WAZIM', 'WAREA', ],
        'DEMAND_proxy': 'GAREA',
        'GSTAT': ['1004',],                 # GSTAT - 1004: only existing, fully constructed buildings
        'GKLAS': ['1110','1121','1276'],    # GKLAS - 1110: only 1 living space per building; 1121: Double-, row houses with each appartment (living unit) having it's own roof; 1276: structure for animal keeping (most likely still one owner)
        'GBAUJ_minmax': [1950, 2022],       # GBAUJ_minmax: range of years of construction
        'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
        # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                            # GANZWHG - total number of apartments in building
                                            # GAZZI - total number of rooms in building
        },
    'solkat_selection_specs': {
        'col_partition_union': 'SB_UUID',     # column name used for the union of partitions
        'GWR_EGID_buffer_size': 2,            # buffer size in meters for the GWR selection
        }   
    }


def data_aggregation_MASTER(dataagg_settings_func):

    # SETTIGNS --------------------------------------------------------------------
    if not isinstance(dataagg_settings_func, dict):
        print('  USE LOCAL SETTINGS - DICT')
        dataagg_settings = dataagg_settings_local
    else:
        dataagg_settings = dataagg_settings_func


    # SETUP -----------------------------------------------------------------------
    # set working directory
    wd_path = dataagg_settings['wd_path_laptop'] if not dataagg_settings['script_run_on_server'] else dataagg_settings['wd_path_server']
    data_path = f'{wd_path}_data'
        
    # create directory + log file
    preprepd_path = f'{data_path}/output/preprep_data' 
    if not os.path.exists(preprepd_path):
        os.makedirs(preprepd_path)
    log_name = f'{data_path}/output/preprep_data_log.txt'
    total_runtime_start = datetime.now()

    summary_log = f'{data_path}/output/topo_summary_log.txt'
    chapter_to_logfile(f'OptimalPV - Sample Summary of Building Topology', summary_log, overwrite_file=True)
    subchapter_to_logfile(f'data_aggregation_MASTER', summary_log)



    # get bfs numbers from canton selection if applicable
    if not not dataagg_settings['kt_numbers']: 
        dataagg_settings['bfs_numbers'] = auxiliary_functions.get_bfs_from_ktnr(dataagg_settings['kt_numbers'], data_path, log_name)
        print_to_logfile(f' > no. of kt  numbers in selection: {len(dataagg_settings["kt_numbers"])}', log_name)
        print_to_logfile(f' > no. of bfs numbers in selection: {len(dataagg_settings["bfs_numbers"])}', log_name) 

    # add information to dataagg_settings that's relevant for further functions
    dataagg_settings['log_file_name'] = log_name
    dataagg_settings['summary_file_name'] = summary_log
    dataagg_settings['wd_path'] = wd_path
    dataagg_settings['data_path'] = data_path

    chapter_to_logfile(f'start data_aggregation_MASTER', log_name, overwrite_file=True)
    formated_dataagg_settings = format_MASTER_settings(dataagg_settings)
    print_to_logfile(f' > settings: \n{pformat(formated_dataagg_settings)}', log_name)



    # SPLIT DATA AND GEOMETRY ------------------------------------------------------
    # split data and geometry to avoid memory issues when importing data
    if dataagg_settings['split_data_geometry_AND_slow_api']:
        subchapter_to_logfile('pre-prep data: SPLIT DATA GEOMETRY + IMPORT SLOW APIs', log_name)
        split_data_and_geometry(dataagg_settings_def = dataagg_settings)



    # IMPORT SLOW API DATA ---------------------------------------------------------------
    if dataagg_settings['split_data_geometry_AND_slow_api']:
        dataagg_settings['year_range'] = [2015, 2023] 
        subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
        api_electricity_prices_data(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: API GM by EWR MAPPING', log_name)
        api_pvtarif_gm_ewr_Mapping(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: API PVTARIF', log_name)
        api_pvtarif_data(dataagg_settings_def=dataagg_settings)

        subchapter_to_logfile('pre-prep data: API ENTSOE DayAhead', log_name)
        api_entsoe_ahead_elecpri_data(dataagg_settings_def = dataagg_settings)



    # IMPORT API DATA 2 ---------------------------------------------------------------
    if dataagg_settings['reimport_api_data']:
        subchapter_to_logfile('pre-prep data: SQL GWR DATA', log_name)
        sql_gwr_data(dataagg_settings_def=dataagg_settings)
    else:
        print_to_logfile('\n\n', log_name)
        checkpoint_to_logfile('use already downloaded data on electricity prices, GWR, PV Tarifs', log_name)



    # IMPORT LOCAL DATA + SPATIAL MAPPINGS ------------------------------------------
    # transform spatial data to parquet files for faster import and transformation
    pq_dir_exists_TF = os.path.exists(f'{data_path}/output/preprep_data')
    pq_files_rerun = dataagg_settings['rerun_localimport_and_mappings']

    if not pq_dir_exists_TF or pq_files_rerun:
        subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', log_name)
        # local_data_to_parquet_AND_create_spatial_mappings_bySBUUID(dataagg_settings_def = dataagg_settings)
        local_data_AND_spatial_mappings(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: IMPORT DEMAND TS + match series HOUSES', log_name)
        import_demand_TS_AND_match_households(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('pre-prep data: IMPORT METEO SUNSHINE TS', log_name)
        import_meteo_data(dataagg_settings_def = dataagg_settings)

    else: 
        print_to_logfile('\n', log_name)
        checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)



    # EXTEND WITH TIME FIXED DATA ---------------------------------------------------------------
    cost_df_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/pvinstcost.parquet')
    reextend_fixed_data = dataagg_settings['reextend_fixed_data']

    if not cost_df_exists_TF or reextend_fixed_data:
        subchapter_to_logfile('extend data: ESTIM PV INSTALLTION COST FUNCTION,  ', log_name)
        estimate_pv_cost(dataagg_settings_def = dataagg_settings)

        subchapter_to_logfile('extend data: GET ANGLE+TILT FACTOR + NODE MAPPING', log_name)
        
        get_fake_gridnodes(dataagg_settings_def = dataagg_settings)
        
    
    # -----------------------------------------------------------------------------
    # END 
    chapter_to_logfile(f'END data_aggregation_MASTER\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)

    if not dataagg_settings['script_run_on_server']:
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 1000)

    
    # COPY & RENAME AGGREGATED DATA FOLDER ---------------------------------------------------------------
    # > not to overwrite completed preprep folder while debugging 

    if not os.path.exists(f'{data_path}/output/{dataagg_settings["name_dir_export"]}'):
        dir_data_moveto = f'{data_path}/output/{dataagg_settings["name_dir_export"]}'
        os.makedirs(dir_data_moveto)
    else:
        today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dir_data_moveto = f'{data_path}/output/{dataagg_settings["name_dir_export"]}_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
        if not os.path.exists(dir_data_moveto):
            os.makedirs(dir_data_moveto)
        elif os.path.exists(dir_data_moveto):
            shutil.rmtree(dir_data_moveto)
            os.makedirs(dir_data_moveto)

    file_to_move = glob.glob(f'{data_path}/output/preprep_data/*')
    for f in file_to_move
        if os.path.isfile(f):
            shutil.copy(f, dir_data_moveto)
        elif os.path.isdir(f):
            shutil.copytree(f, f'{dir_data_moveto}/{f.split("/")[-1]}')
    shutil.copy(glob.glob(f'{data_path}/output/preprep_data_log.txt')[0], f'{dir_data_moveto}/preprep_data_log_{dataagg_settings["name_dir_export"]}.txt')
    shutil.copy(glob.glob(f'{data_path}/output/topo_summary_log.txt')[0], f'{dir_data_moveto}/topo_summary_log_{dataagg_settings["name_dir_export"]}.txt')

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


