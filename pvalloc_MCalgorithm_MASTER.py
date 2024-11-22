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

    import pv_allocation.alloc_algorithm as algo
    import pv_allocation.inst_selection as select



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
        show_debug_prints = pvalloc_settings['show_debug_prints']

        total_runtime_start = datetime.now()

    
    # MONTE CARLO ITERATION LOOP ================================================================================
    chapter_to_logfile(f'start pvalloc_MCalgorithm_MASTER for : {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
    print_to_logfile(f'*model allocation specifications*:', log_name)
    print_to_logfile(f'> n_bfs_municipalities: {len(pvalloc_settings["bfs_numbers"])} \n> n_months_prediction: {pvalloc_settings["months_prediction"]} \n> n_montecarlo_iterations: {pvalloc_settings["MC_loop_specs"]["montecarlo_iterations"]}', log_name)
 

    # CREATE MC DIR + TRANSFER INITIAL DATA FILES ----------------------------------------------

    # pvalloc_settings["name_dir_export"] = 'pvalloc_DEV_12m_pvmethod2_selfconsum00'
    montecarlo_iterations = [*range(1, pvalloc_settings['MC_loop_specs']['montecarlo_iterations']+1, 1)]
    fresh_initial_files = pvalloc_settings['MC_loop_specs']['fresh_initial_files']
    rand_seed = pvalloc_settings['algorithm_specs']['rand_seed']
    safety_counter_max = pvalloc_settings['algorithm_specs']['while_inst_counter_max']
    
    # get all initial files to start a fresh MC iteration
    fresh_initial_paths = [f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/{file}' for file in fresh_initial_files]
    topo_time_paths = glob.glob(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/topo_time_subdf/*.parquet')
    all_initial_paths = fresh_initial_paths + topo_time_paths

    max_digits = len(str(max(montecarlo_iterations)))
    # mc_iter = montecarlo_iterations[0]
    # if True:    
    for mc_iter in montecarlo_iterations:
        mc_iter_start = datetime.now()
        subchapter_to_logfile(f'START MC{mc_iter:0{max_digits}} iteration', log_name)
        # print_to_logfile(f'\n-- START MC{mc_iter:0{max_digits}} iteration  {25*"-"}', log_name)

        # copy all initial files to MC directory
        mc_data_path = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/zMC_{mc_iter:0{max_digits}}'
        os.makedirs(mc_data_path, exist_ok=True)
        for file in all_initial_paths:
            shutil.copy(file, mc_data_path)
            
        # remove old files to avoid concatenating old files to iteration-by-iteration interim saves
            # for df_type in ['npv_df', 'pred_inst_df']:
            #     df_paths = glob.glob(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}/{df_type}.*')
            #     for f in df_paths:
            #         os.remove(f)


        # ALLOCATION ALGORITHM ----------------------------------------------    
        dfuid_installed_list = []
        pred_inst_df = pd.DataFrame()  
        months_prediction_df = pd.read_parquet(f'{mc_data_path}/months_prediction.parquet')
        months_prediction = months_prediction_df['date']
        constrcapa = pd.read_parquet(f'{mc_data_path}/constrcapa.parquet')
        
        for i_m, m in enumerate(months_prediction):
            # print_to_logfile(f'\n-- MC{mc_iter:0{max_digits}} -- allocation month: {m} --', log_name)
            print_to_logfile(f'-- month {m} -- iter MC{mc_iter:0{max_digits}} -- ', log_name)
            start_allocation_month = datetime.now()
            i_m = i_m + 1        


            # GRID PREM UPDATE ==========
            algo.update_gridprem(pvalloc_settings, mc_data_path, m, i_m)

            # NPV UPDATE ==========
            npv_df = algo.update_npv_df(pvalloc_settings, mc_data_path, m, i_m)


            # initialize constr capacity ----------
            constr_built_m = 0
            if m.year != (m-1).year:
                constr_built_y = 0
            constr_capa_m = constrcapa.loc[constrcapa['date'] == m, 'constr_capacity_kw'].iloc[0]
            constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()
            
            
            # INSTALLATION PICK ==========
            safety_counter = 0
            print_to_logfile(f'run installation pick while loop', log_name)
            while( (constr_built_m <= constr_capa_m) & (constr_built_y <= constr_capa_y) & (safety_counter <= safety_counter_max) ):
                
                if npv_df.shape[0] == 0:
                    checkpoint_to_logfile(f' npv_df is EMPTY, exit while loop', log_name, 1, show_debug_prints)
                    safety_counter = safety_counter_max

                if npv_df.shape[0] > 0: 
                    # checkpoint_to_logfile(f' npv_df with 0 < rows, select inst and adjust topology', log_name, 1, show_debug_prints)
                    inst_power = select.select_AND_adjust_topology(pvalloc_settings, 
                                                    mc_data_path,
                                                    dfuid_installed_list, 
                                                    pred_inst_df,
                                                    i_m, m)
                    
                # Adjust constr_built capacity ----------
                constr_built_m, constr_built_y, safety_counter = constr_built_m + inst_power, constr_built_y + inst_power, safety_counter + 1

                # State Loop Exit ----------
                overshoot_rate = pvalloc_settings['constr_capacity_specs']['constr_capa_overshoot_fact']
                constr_m_TF, constr_y_TF, safety_TF = constr_built_m > constr_capa_m*overshoot_rate, constr_built_y > constr_capa_y*overshoot_rate, safety_counter > safety_counter_max


                if any([constr_m_TF, constr_y_TF, safety_TF]):
                    print_to_logfile(f'exit While Loop', log_name)
                    if constr_m_TF:
                        checkpoint_to_logfile(f'exceeded constr_limit month (constr_m_TF:{constr_m_TF}), {round(constr_built_m,1)} of {round(constr_capa_m,1)} kW capacity built', log_name, 1, show_debug_prints)
                    if constr_y_TF:
                        checkpoint_to_logfile(f'exceeded constr_limit year (constr_y_TF:{constr_y_TF}), {round(constr_built_y,1)} of {round(constr_capa_y,1)} kW capacity built', log_name, 1, show_debug_prints)
                    if safety_TF:
                        checkpoint_to_logfile(f'exceeded safety counter (safety_TF:{safety_TF}), {safety_counter} rounds for safety counter max of: {safety_counter_max}', log_name, 1, show_debug_prints)

                    if constr_m_TF or constr_y_TF:    
                        checkpoint_to_logfile(f'{safety_counter} pv installations allocated', log_name, 3, show_debug_prints)                    
                    # safety_counter = 0

            checkpoint_to_logfile(f'end month allocation, runtime: {datetime.now() - start_allocation_month} (hh:mm:ss.s)', log_name, 1, show_debug_prints)

    
        # CLEAN UP interim files of MC run ----------
        topo_time_paths = glob.glob(f'{mc_data_path}/topo_subdf_*.parquet')
        for f in topo_time_paths:
            os.remove(f)

        mc_iter_end = datetime.now()
        mc_iter_time = mc_iter_end - mc_iter_start
        subchapter_to_logfile(f'END MC{mc_iter:0{max_digits}}, runtime: {mc_iter_time} (hh:mm:ss.s)', log_name)
        print_to_logfile(f'\n', log_name)
        # print_to_logfile(f'\n-- END MC{mc_iter:0{max_digits}} iteration  {25*"-"}\n-- runtime: () --\n\n', log_name


    # END  ================================================================
    chapter_to_logfile(f'END pvalloc_MCalgorithmn_MASTER\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)
    if not pvalloc_settings['script_run_on_server']:
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 300)
        winsound.Beep(1000, 1000)

    # COPY & RENAME PVALLOC DATA FOLDER ---------------------------------------------------------------
    # > not to overwrite completed folder while debugging 
    dir_alloc_moveto = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}'

    shutil.copy(glob.glob(f'{data_path}/output/pvalloc_MCalgo_log.txt')[0], f'{dir_alloc_moveto}/pvalloc_MCalgo_log_{pvalloc_settings["name_dir_export"]}.txt')

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


