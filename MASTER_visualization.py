# -----------------------------------------------------------------------------
# MASTER_visualization.py
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 



# PACKAGES --------------------------------------------------------------------
if True:
    import os as os
    import sys
    sys.path.append(os.getcwd())

    # external packages
    import os as os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import json 
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.colors as pc
    import copy
    import glob
    import matplotlib.pyplot as plt
    import winsound
    import itertools
    import shutil
    import scipy.stats as stats

    from datetime import datetime
    from pprint import pformat
    from shapely.geometry import Polygon, MultiPolygon
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde
    

    # own packages and functions
    import pv_allocation.default_settings as pvalloc_default_sett
    import visualizations.defaults_settings as visual_default_sett
    
    from auxiliary_functions import *
    
    import visualizations.plot_ind_var_summary_stats as plot_ind_var_summary_stats
    import visualizations.plot_ind_hist_pvcapaprod_sanitycheck as plot_ind_hist_pvcapaprod_sanitycheck
    import visualizations.plot_ind_boxp_radiation_rng_sanitycheck as plot_ind_boxp_radiation_rng_sanitycheck
    import visualizations.plot_ind_charac_omitted_gwr as plot_ind_charac_omitted_gwr
    import visualizations.plot_ind_line_meteo_radiation as plot_ind_line_meteo_radiation
    import visualizations.plot_ind_line_installedCap as plot_ind_line_installedCap
    import visualizations.plot_ind_line_PVproduction as plot_ind_line_PVproduction
    import visualizations.plot_ind_line_productionHOY_per_node as plot_ind_line_productionHOY_per_node
    import visualizations.plot_ind_line_gridPremiumHOY_per_node as plot_ind_line_gridPremiumHOY_per_node
    import visualizations.plot_ind_line_gridPremium_structure as plot_ind_line_gridPremium_structure
    import visualizations.plot_ind_hist_NPV_freepartitions as plot_ind_hist_NPV_freepartitions
    import visualizations.plot_ind_lineband_contcharact_newinst as plot_ind_lineband_contcharact_newinst
    
    import visualizations.plot_ind_map_topo_egid as plot_ind_map_topo_egid
    import visualizations.plot_ind_map_node_connections as plot_ind_map_node_connections
    import visualizations.plot_ind_map_omitted_egids as plot_ind_map_omitted_egids


    # import visualizations.plot_mc_line_PVproduction as plot_mc_line_PVproduction



def MASTER_visualization(pvalloc_scenarios_func, visual_settings_func):
    # SETTINGS ------------------------------------------------------------------------------------------------------
    if True:
        if not isinstance(pvalloc_scenarios_func, dict):
            print(' USE LOCAL SETTINGS - DICT  ')
            pvalloc_scenarios = pvalloc_default_sett.get_default_pvalloc_settings()
        else:
            pvalloc_scenarios = pvalloc_scenarios_func

        if not isinstance(visual_settings_func, dict) or visual_settings_func == {}:
            visual_settings = visual_default_sett.get_default_visual_settings()
        else:
            visual_settings = visual_settings_func

        pvalloc_sett_run_on_server = pvalloc_scenarios.get(next(iter(pvalloc_scenarios))).get('script_run_on_server')

    # SETUP ------------------------------------------------------------------------------------------------------
    if True: 
        # general setup for paths etc.
        # first_alloc_sett = pvalloc_scenarios[list(pvalloc_scenarios.keys())[0]]
        # wd_path = first_alloc_sett['wd_path_laptop'] if not first_alloc_sett['script_run_on_server'] else first_alloc_sett['wd_path_server']
        wd_path = os.getcwd()
        data_path = f'{wd_path}_data'

        # create directory + log file
        visual_path = f'{data_path}/visualizations'
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)

        log_name = f'{visual_path}/visual_log.txt'
        total_runtime_start = datetime.now()


        # extract scenario settings + information ------------------------
        scen_dir_export_list, pvalloc_scen_list = [], []
        # scen_dir_import_list, T0_prediction_list, months_prediction_list = [], [], [], [] T0_prediction_list, months_lookback_list, months_prediction_list = [], [], [] pvalloc_scen_list = []
        for key, val in pvalloc_scenarios.items():
            pvalloc_settings_path = glob.glob(f'{data_path}/output/{key}/pvalloc_settings.json')
            # pvalloc_settings_path = glob.glob(f'{data_path}/pvalloc_runs/{key}/pvalloc_settings.json')
            
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
    

        # create directory for plots by scen ----------------
        for key, val in pvalloc_scenarios.items():
            scen = val['name_dir_export']
            # scen = key
            scen_path = f'{data_path}/visualizations/{scen}'
            
            if os.path.exists(scen_path):
                n_same_names = len(glob.glob(f'{scen_path}*/'))
                old_dir_rename = f'{scen_path} ({n_same_names})'
                os.rename(scen_path, old_dir_rename)

            os.makedirs(scen_path)

        # if visual_settings['remove_previous_plots']:
        #     all_html = glob.glob(f'{data_path}/output/visualizations/*.html')
        #     for f in all_html:
        #         os.remove(f)

        if visual_settings['remove_old_plot_scen_directories']:
            old_plot_scen_dirs = glob.glob(f'{data_path}/visualizations/*(*)')
            for dir in old_plot_scen_dirs:
                try:    
                    shutil.rmtree(dir)
                except:
                    print(f'Could not remove {dir}')

    chapter_to_logfile(f'start MASTER_visualization\n', log_name, overwrite_file=True)




    # PLOT IND SCEN: pvalloc_initalization + sanitycheck ------------------------------------------------------------------------------------------------------


    # plot ind - var: summary statistics --------------------
    plot_ind_var_summary_stats.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - hist: sanity check capacity & production --------------------
    plot_ind_hist_pvcapaprod_sanitycheck.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - hist: radiation range --------------------
    plot_ind_boxp_radiation_rng_sanitycheck.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - var: disc charac omitted gwr_egids --------------------
    plot_ind_charac_omitted_gwr.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)


    # plot ind - line: meteo radiation over time --------------------
    plot_ind_line_meteo_radiation.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)



    # PLOT IND SCEN: pvalloc_MC_algorithm ------------------------------------------------------------------------------------------------------

    
    # plot ind - line: Installed Capacity per Month & per BFS --------------------
    plot_ind_line_installedCap.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - hist: pv production deviation --------------------
    plot_ind_line_PVproduction.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)


    # plot ind - line: Production + Feedin HOY per Node --------------------
    plot_ind_line_productionHOY_per_node.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)   
    

    # plot ind - line: Grid Premium per Hour of Year --------------------
    plot_ind_line_gridPremiumHOY_per_node.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)


    # 
    plot_ind_line_gridPremium_structure.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)


    # plot ind - hist: NPV possible PV inst before / after --------------------
    plot_ind_hist_NPV_freepartitions.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)
    


    # map ind - topo_egid --------------------
    plot_ind_map_topo_egid.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name, )

    # map ind - node_connections --------------------
    plot_ind_map_node_connections.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name,)

    # map ind - omitted gwr_egids --------------------
    plot_ind_map_omitted_egids.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name, )


    # plot ind - lineband: continuous characteristics for new installations --------------------
    plot_ind_lineband_contcharact_newinst.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)




    # PLOT IND SCEN: aggregated pvalloc_MC_algorithm ------------------------------------------------------------------------------------------------------

    # plot mc - line: PV production ----------------
    # plot_mc_line_PVproduction.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)





    # END  ================================================================
    chapter_to_logfile(f'END MASTER_visualization\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)
    if not pvalloc_sett_run_on_server:
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 1000)

