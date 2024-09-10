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

    import glob
    import shutil
    import winsound
    import subprocess
    import pprint

    # own packages and functions
    import auxiliary_functions
    from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr, format_MASTER_settings
    import pv_allocation.initialization as  initial
    import pv_allocation.alloc_algorithm as algo
    import pv_allocation.topo_visualization as visual
    import pv_allocation.inst_selection as select

    from pv_allocation.initialization import *
    from pv_allocation.alloc_algorithm import *
    from pv_allocation.inst_selection import *
    from pv_allocation.topo_visualization import *

pvalloc_settings_local = {
    'name_dir_export': 'pvalloc_BL_smallsample',              # name of the directory where all proccessed data is stored at the end of the code file 
    'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h', # name of the directory where preprepared data is stored and accessed by the code
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
    'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

    'kt_numbers': [], # [13, ][11,12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain indiviual municipalities
    'bfs_numbers': [2791, 2784, 2781, 2789, 2782, 2793, ], #2787, 2792, 2613, 2614, 2476, 2477],
    'T0_prediction': '2023-01-01 00:00:00', 
    'months_lookback': 12*1,
    'months_prediction': 5,

    'script_run_on_server':     False,                           # F: run on private computer, T: run on server
    'show_debug_prints':        True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
    'fast_debug_run':           False,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
    'n_egid_in_topo': 200, 
    'recreate_topology':            True, 
    'recalc_economics_topo_df':     True,
    'run_allocation_loop':          True,

    'create_map_of_topology':       False,

    'recalc_npv_all_combinations':  True,
    'test_faster_if_subdf_deleted': False,
    'test_faster_npv_update_w_subdf_npry': True, 

    'algorithm_specs': {
        'inst_selection_method': 'prob_weighted_npv', # random, prob_weighted_npv, 
        'montecarlo_iterations': 1,
        'keep_files_each_iterations': ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 
        'keep_files_only_one': ['elecpri.parquet', 'pvtarif.parquet', 'pv.parquet', 'meteo_ts'],
        'rand_seed': 42,                            # random seed set to int or None
        'while_inst_counter_max': 5000,
        'capacity_tweak_fact': 1, 
        'topo_subdf_partitioner': 800,
    },
    'gridprem_adjustment_specs': {
        'voltage_assumption': '',
        'tier_description': 'tier_level: (voltage_threshold, gridprem_plusRp_kWh)',
        'colnames': ['tier_level', 'vltg_threshold', 'gridprem_plusRp_kWh'],
        'tiers': { 
            1: [200, 1], 
            2: [400, 3],
            4: [600, 7],
            5: [800, 15], 
            6: [1500, 50],
            },},
        # 'tiers': { 
        #     1: [2000, 1], 
        #     2: [4000, 3],
        #     4: [6000, 7],
        #     5: [8000, 15], 
        #     6: [15000, 50],
        #     },},
    'tech_economic_specs': {
        'self_consumption_ifapplicable': 1,
        'interest_rate': 0.01,
        'pvtarif_year': 2022, 
        'pvtarif_col': ['energy1', 'eco1'],
        'elecpri_year': 2022,
        'elecpri_category': 'H8', 
        'invst_maturity': 25,
        'conversion_m2tokW': 0.1,  # A 1m2 area can fit 0.1 kWp of PV Panels
    },
    'weather_specs': {
        'meteoblue_col_radiation_proxy': 'Basel Direct Shortwave Radiation',
        'weather_year': 2022,
    },
    'constr_capacity_specs': {
        'ann_capacity_growth': 0.1,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
        'summer_months': [4,5,6,7,8,9,],
        'winter_months': [10,11,12,1,2,3,],
        'share_to_summer': 0.6, 
        'share_to_winter': 0.4,
    },
    'gwr_selection_specs': {
                'solkat_max_n_partitions': 10,
                'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                            'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                            'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
                'dwelling_cols': None, # ['EGID', 'WAZIM', 'WAREA', ],
                'DEMAND_proxy': 'GAREA',
                'GSTAT': ['1004',],                 # GSTAT - 1004: only existing, fully constructed buildings
                'GKLAS': ['1110','1121','1276',],                 # GKLAS - 1110: only 1 living space per building
                'GBAUJ_minmax': [1950, 2022],       # GBAUJ_minmax: range of years of construction
                # 'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
                # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                                    # GANZWHG - total number of apartments in building
                                                    # GAZZI - total number of rooms in building
            },
}


def pv_allocation_MASTER(pvalloc_settings_func):

    # SETTIGNS --------------------------------------------------------------------
    if not isinstance(pvalloc_settings_func, dict):
        print(' USE LOCAL SETTINGS - DICT  ')
        pvalloc_settings = pvalloc_settings_local
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
        log_name = f'{data_path}/output/pvalloc_log.txt'
        total_runtime_start = datetime.now()


        # extend settings dict with relevant informations for later functions
        if not not pvalloc_settings['kt_numbers']:
            pvalloc_settings['bfs_numbers'] = auxiliary_functions.get_bfs_from_ktnr(pvalloc_settings['kt_numbers'], data_path, log_name)
            print_to_logfile(f' > no. of kt  numbers in selection: {len(pvalloc_settings["kt_numbers"])}', log_name)
            print_to_logfile(f' > no. of bfs numbers in selection: {len(pvalloc_settings["bfs_numbers"])}', log_name) 

        elif (not pvalloc_settings['kt_numbers']) and (not not pvalloc_settings['bfs_numbers']):
            pvalloc_settings['bfs_numbers'] = [str(bfs) for bfs in pvalloc_settings['bfs_numbers']]

        pvalloc_settings['log_file_name'] = log_name
        pvalloc_settings['wd_path'] = wd_path
        pvalloc_settings['data_path'] = data_path
        pvalloc_settings['pvalloc_path'] = pvalloc_path
        interim_path = get_interim_path(pvalloc_settings)
        pvalloc_settings['interim_path'] = interim_path
        show_debug_prints = pvalloc_settings['show_debug_prints']
        conv_m2toKWP = pvalloc_settings['tech_economic_specs']['conversion_m2tokW']

    chapter_to_logfile(f'start pv_allocation_MASTER for: {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
    formated_pvalloc_settings = format_MASTER_settings(pvalloc_settings)
    print_to_logfile(f'pvalloc_settings: \n{pformat(formated_pvalloc_settings)}', log_name)



    # INITIALIZATION ================================================================
    if pvalloc_settings['recreate_topology']:
        subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', log_name)
        topo, df_list, df_names = initial.import_prepre_AND_create_topology(pvalloc_settings)

    elif not pvalloc_settings['recreate_topology']:
        subchapter_to_logfile('initialization: IMPORT EXISITNG TOPOLOGY', log_name) 
        df_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_egid', 'Map_egid_demandtypes', 'pv', 'pvtarif', 'elecpri', 'angle_tilt_df', 'Map_egid_nodes']
        topo, df_list, df_names = initial.import_exisitng_topology(pvalloc_settings, df_search_names= df_names)
        shutil.copy(f'{data_path}/output/pvalloc_run/topo_egid.json', f'{data_path}/output/pvalloc_run/topo_egid_before_alloc.json')

    subchapter_to_logfile('initialization: IMPORT TS DATA', log_name)
    ts_list, ts_names = initial.import_ts_data(pvalloc_settings)

    subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', log_name)
    constrcapa, months_prediction, months_lookback = define_construction_capacity(pvalloc_settings, topo, df_list, df_names, ts_list, ts_names)



    # PREPARE TOPO_TIME SPECIFIC DFs ================================================================

    # CALC ECONOMICS for TOPO_DF ----------------------------------------------------------------
    if pvalloc_settings['recalc_economics_topo_df']:
        subchapter_to_logfile('prep: CALC ECONOMICS for TOPO_DF', log_name)
        # groupby_cols_topoaggdf = ['EGID', 'df_uid', 'grid_node', 'bfs', 'gklas', 'demandtype',
        #                 'pvinst_TF', 'pvsource', 'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
        #                 'FLAECHE', 'AUSRICHTUNG', 'STROMERTRAG', 'NEIGUNG', 'angletilt_factor']
        # agg_cols_name_topoaggdf = ['pvprod_kW', 'demand_kW', 'selfconsum_kW', 'netdemand_kW', 'netfeedin_kW', 'econ_inc_chf', 'econ_spend_chf']
        # agg_cols_method_topoaggdf = ['sum', 'sum', 'sum', 'sum', 'sum', 'sum', 'sum']

        # topo_df, topo_agg_df = algo.calc_economics_in_topo_df(pvalloc_settings, topo, 
        #                                                       groupby_cols_topoaggdf, agg_cols_name_topoaggdf, agg_cols_method_topoaggdf, 
        #                                                       df_list, df_names, ts_list, ts_names)
        algo.calc_economics_in_topo_df(pvalloc_settings, topo, 
                                        df_list, df_names, ts_list, ts_names)

    if False: #not pvalloc_settings['recalc_economics_topo_df']:
        # topo_agg_df_in_pvallocrun = os.path.exists(f'{data_path}/output/pvalloc_run/topo_agg_df.parquet')
        # topo_agg_df_in_interim = os.path.exists(f'{interim_path}/topo_agg_df.parquet')
        # topo_agg_df_path = f'{interim_path}/topo_agg_df.parquet' if topo_agg_df_in_interim else f'{data_path}/output/pvalloc_run/topo_agg_df.parquet'

        # subchapter_to_logfile('allocation algorithm: GET TOPO_DF from pvalloc or interim_path', log_name)
        # topo_agg_df = pd.read_parquet(topo_agg_df_path)
        # NOTE: Remove if statement if no longer necessary
        # if os.path.exists(f'{topo_agg_df_path}/topo_df.parquet'):
            # topo_df = pd.read_parquet(f'{topo_agg_df_path}/topo_df.parquet')
        print('all no longer necessary')


    # CREATE MAP OF TOPO_DF ----------------------------------------------------------------
    if pvalloc_settings['create_map_of_topology']:
        subchapter_to_logfile('visualization: CREATE MAP OF TOPOLOGY_DF', log_name)
        visual.create_map_of_topology(pvalloc_settings)


    # ALLOCATION ALGORITHM ----------------------------------------------                   
    if False:   #pvalloc_settings['recalc_npv_all_combinations']:
        subchapter_to_logfile('allocation algorithm: NPV CALCULATION for ALL COMBINATIONS', log_name)
        # npv_df = algo.calc_npv_partition_combinations(pvalloc_settings, topo_agg_df)  
    elif False:  # not pvalloc_settings['recalc_npv_all_combinations']:
        subchapter_to_logfile('allocation algorithm: IMPORT NPV_DF', log_name)
        # npv_df_in_pvallocrun = os.path.exists(f'{data_path}/output/pvalloc_run/npv_df.parquet')
        # npv_df_in_interim = os.path.exists(f'{interim_path}/npv_df.parquet')
        # npv_path = f'{interim_path}/npv_df.parquet' if npv_df_in_interim else f'{data_path}/output/pvalloc_run/npv_df.parquet'

        # npv_df = pd.read_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')
    #
    if pvalloc_settings['run_allocation_loop']: 
        subchapter_to_logfile('allocation algorithm: START LOOP FOR PRED MONTH', log_name)

        months_lookback = pvalloc_settings['months_lookback']
        rand_seed = pvalloc_settings['algorithm_specs']['rand_seed']
        safety_counter_max = pvalloc_settings['algorithm_specs']['while_inst_counter_max']
        
        # aggregation cols for npv update
        groupby_cols_topoaggdf = ['EGID', 'df_uid', 'grid_node', 'bfs', 'gklas', 'demandtype',
                        'pvinst_TF', 'pvsource', 'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
                        'FLAECHE', 'AUSRICHTUNG', 'STROMERTRAG', 'NEIGUNG', 'angletilt_factor']
        agg_cols_topoaggdf = {'FLAECH_angletilt': 'sum','pvprod_kW': 'sum', 
                            'demand_kW': 'sum', 'selfconsum_kW': 'sum', 
                            'netdemand_kW': 'sum', 'netfeedin_kW': 'sum', 
                            'econ_inc_chf': 'sum', 'econ_spend_chf': 'sum'}

        # empty lists and dfs for aggregation later
        dfuid_installed_list = []
        pred_inst_df = pd.DataFrame()

        # create interim_predictions folder to see month by month changes
        if not os.path.exists(f'{data_path}/output/pvalloc_run/interim_predictions'):
            os.makedirs(f'{data_path}/output/pvalloc_run/interim_predictions')
        else:
            old_files = glob.glob(f'{data_path}/output/pvalloc_run/interim_predictions/*')
            for f in old_files:
                os.remove(f)



    # ALLOCATION LOOP ----------------------------------------------
        for i, m in enumerate(months_prediction):
            print_to_logfile(f'\n-- Allocation for month: {m} {25*"-"}', log_name)
            start_allocation_month = datetime.now()


            # GRID PREM UPDATE ==========
            algo.update_gridprem(pvalloc_settings,
                        df_list, df_names,
                        ts_list, ts_names, 
                        m)
            

            # NPV UPDATE ==========
            npv_df = algo.update_npv_df(pvalloc_settings,
                        groupby_cols_topoaggdf, agg_cols_topoaggdf, 
                        df_list, df_names,
                        ts_list, ts_names,)


            # initialize constr capacity ----------
            constr_built_m = 0
            if m.year != (m-1).year:
                constr_built_y = 0
            constr_capa_m = constrcapa.loc[constrcapa['date'] == m, 'constr_capacity_kw'].iloc[0]
            constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()


            # INSTALLATION PICK ==========
            safety_counter = 0
            while( (constr_built_m < constr_capa_m) & (constr_built_y < constr_capa_y) & (safety_counter < safety_counter_max) ):
                
                if npv_df.shape[0] == 0:
                    checkpoint_to_logfile(f' npv_df is EMPTY, exit while loop', log_name, 1, show_debug_prints)
                    safety_counter = safety_counter_max

                if npv_df.shape[0] > 0: 
                    inst_power = select.select_AND_adjust_topology(pvalloc_settings, npv_df, 
                                                    pvalloc_settings['algorithm_specs']['inst_selection_method'],
                                                    dfuid_installed_list, 
                                                    pred_inst_df,
                                                    m)

                    # Adjust constr_built capacity----------
                    constr_built_m += inst_power
                    constr_built_y += inst_power
                    safety_counter += 1

                    # State Loop Exit ----------
                    constr_m_TF  = constr_built_m >= constr_capa_m
                    constr_y_TF  = constr_built_y >= constr_capa_y
                    safety_TF = safety_counter >= safety_counter_max

                    if safety_counter % 50 == 0:
                        print(f'PRINT_SAFETY_COUNTER: {safety_counter}')

                if any([constr_m_TF, constr_y_TF, safety_TF]):
                    checkpoint_to_logfile(f' Exit While Loop, constraint met : constr_m_TF: {constr_m_TF}, constr_y_TF: {constr_y_TF}, safety_TF: {safety_TF}', log_name, 1, True)
                    checkpoint_to_logfile(f' {safety_counter} pv installations allocated', log_name, 3, show_debug_prints)
                    safety_counter = 0

            checkpoint_to_logfile(f'end month allocation, runtime: {datetime.now() - start_allocation_month} (hh:mm:ss.s)', log_name, 1, show_debug_prints)



    # COPY & RENAME AGGREGATED DATA FOLDER ---------------------------------------------------------------
    # > not to overwrite completed preprep folder while debugging 

    # -----------------------------------------------------------------------------
    # END 
    chapter_to_logfile(f'END pv_allocation_MASTER\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)

    if not pvalloc_settings['script_run_on_server']:
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 1000)
    # -----------------------------------------------------------------------------



    if  not os.path.exists(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}'):
        dir_alloc_moveto = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}'
        os.makedirs(dir_alloc_moveto)
    else:
        today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dir_alloc_moveto = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
        if not os.path.exists(dir_alloc_moveto):
            os.makedirs(dir_alloc_moveto)
        elif os.path.exists(dir_alloc_moveto):
            shutil.rmtree(dir_alloc_moveto)
            os.makedirs(dir_alloc_moveto)

    file_to_move = glob.glob(f'{data_path}/output/pvalloc_run/*')
    for f in file_to_move:
        if os.path.isfile(f):
            shutil.copy(f, dir_alloc_moveto)
        elif os.path.isdir(f):
            shutil.copytree(f, os.path.join(dir_alloc_moveto, os.path.basename(f)))
    shutil.copy(glob.glob(f'{data_path}/output/pvalloc_log.txt')[0], f'{dir_alloc_moveto}/pvalloc_log_{pvalloc_settings["name_dir_export"]}.txt')
        



# MONTE CARLO LOOP ----------------------------------------------
"""

    # MONTE CARLO PREP ----------------------------------------------
    mc_iter = pvalloc_settings['algorithm_specs']['montecarlo_iterations']
    keep_mciter_files_list = pvalloc_settings['algorithm_specs']['keep_files_each_iterations']
    keep_onefile_list = pvalloc_settings['algorithm_specs']['keep_files_only_one']
    name_dir_export = pvalloc_settings['name_dir_export']

    
    mc_output_path = f'{data_path}/output/{name_dir_export}_mc_model'

    # keep all iteration files
    if not os.path.exists(mc_output_path):
        os.makedirs(mc_output_path)
    else:
        old_files = glob.glob(f'{mc_output_path}/*')
        for f in old_files:
            os.remove(f)


    # MONTE CARLO LOOP ===============================================
    for i_mc in range(1, mc_iter+1):
"""

# EXPORT MC FILES ----------------------------------------------
"""
        # EXPORT FILES ----------------------------------------------
        os.makedirs(f'{mc_output_path}/mc_iterations')
        for find_file in keep_mciter_files_list:
            f_name = find_file.split('.')[0]
            f_suffix = find_file.split('.')[1]
            shutil.move(f'{data_path}/output/pvalloc_run/{find_file}', f'{mc_output_path}/mc_iterations/{f_name}_mc{i_mc}.{f_suffix}')
        if os.path.exists(f'{data_path}/output/pvalloc_log.txt'):
            shutil.move(f'{data_path}/output/pvalloc_log.txt', f'{mc_output_path}/mc_iterations/pvalloc_log_mc{i_mc}.txt')

        for find_file in keep_onefile_list:
            shutil.move(f'{data_path}/output/pvalloc_run/{find_file}', f'{mc_output_path}/{find_file}')
"""

