import os

import data_aggregation_MASTER
import pvalloc_initialization_MASTER
import pvalloc_MCalgorithm_MASTER
#mport pvalloc_postprocessing_MASTER
import visualization_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
months_pred = 12 #600 #36
run_on_server = False
bfs_numbers = [2791, 2787 ]#, 2792, 2784, 2793, 2782, 2781,]

run_dataagg =   False
run_alloc =     True
run_visual =    False


# data_aggregation 
dataagg_scenarios = {   
    # 'preprep_BLBSSO_18to22_1and2homes_API_reimport':{
    #     'script_run_on_server': run_on_server, 
    #     'kt_numbers': [13,12,11],
    #     'year_range': [2018, 2022], 
    #     'split_data_geometry_AND_slow_api': True, 
    #     'gwr_selection_specs': {'GKLAS': ['1110','1121',],}, 
    # },
    #  
    'preprep_BL_20to22_1and2homes_buff002':{
        'script_run_on_server': run_on_server, 
        # 'kt_numbers': [13,], 
        'bfs_numbers': bfs_numbers,
        'year_range': [2020, 2022], 
        'split_data_geometry_AND_slow_api': True, 
        'demand_specs':{
            'input_data_source': "NETFLEX",},
        'gwr_selection_specs': {'GKLAS': ['1110','1121',],}, 
    }, 
}
dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
pvalloc_scenarios={
    f'pvalloc_DEV_{months_pred}m_meth3_selfcon00_DirDiffRad':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': bfs_numbers,
        'create_gdf_export_of_topology':    True,
        'recalc_economics_topo_df':         True,

        'tech_economic_specs': {
            'self_consumption_ifapplicable': 0,
            'pvprod_calc_method': 'method3',},
        'algorithm_specs': {
            'rand_seed': 42,},
        'weather_specs': {
            'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
        },

    # f'pvalloc_DEV_{months_pred}m_meth3_selfcon00_DirDiffRad':{
    #     'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     'bfs_numbers': bfs_numbers,
    #     # 'create_gdf_export_of_topology':    True,
    #     'tech_economic_specs': {
    #         'self_consumption_ifapplicable': 0,
    #         'pvprod_calc_method': 'method3',},
    #     'algorithm_specs': {
    #         'rand_seed': 42,},
    #     'weather_specs': {
    #         'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
    #     },
    # f'pvalloc_DEV_{months_pred}m_meth2_selfcon00_DirDiffRad':{
    #     'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     'bfs_numbers': bfs_numbers,
    #     'tech_economic_specs': {
    #         'self_consumption_ifapplicable': 0,
    #         'pvprod_calc_method': 'method2',},
    #     'algorithm_specs': {
    #         'rand_seed': 42,},
    #     'weather_specs': {
    #         'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
    #     },
    # f'pvalloc_DEV_{months_pred}m_meth3_selfcon00_DirRad':{
    #     'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     'bfs_numbers': bfs_numbers,
    #     'tech_economic_specs': {
    #         'self_consumption_ifapplicable': 0,
    #         'pvprod_calc_method': 'method3',},
    #     'algorithm_specs': {
    #         'rand_seed': 42,},
    #     'weather_specs': {
    #         'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation',]}
    #     },  
    # f'pvalloc_DEV_{months_pred}m_meth3_selfcon05_DirDiffRad':{
    #     'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     'bfs_numbers': bfs_numbers,
    #     # 'create_gdf_export_of_topology':    True,
    #     'tech_economic_specs': {
    #         'self_consumption_ifapplicable': 0.5,
    #         'pvprod_calc_method': 'method3',},
    #     'algorithm_specs': {
    #         'rand_seed': 42,},
    #     'weather_specs': {
    #         'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
    #     },
  
}

parklplatz = {
    f'pvalloc_DEV_{months_pred}m_meth2_selfcon00_DirDiffRad':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': bfs_numbers,
        'tech_economic_specs': {
            'self_consumption_ifapplicable': 0,
            'pvprod_calc_method': 'method2',},
        'algorithm_specs': {
            'rand_seed': 42,},
        'weather_specs': {
            'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
        },
    f'pvalloc_DEV_{months_pred}m_meth3_selfcon00_DirRad':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': bfs_numbers,
        'tech_economic_specs': {
            'self_consumption_ifapplicable': 0,
            'pvprod_calc_method': 'method3',},
        'algorithm_specs': {
            'rand_seed': 42,},
        'weather_specs': {
            'meteoblue_col_radiation_proxy': ['Basel Direct Shortwave Radiation',]}
        },  
}
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': True,
        'node_selection_for_plots': ['node1', 'node10', 'node15'], # or None for all nodes

        'plot_ind_line_productionHOY_per_node':  True,
        'plot_ind_line_installedCap_per_month':  True,
        'plot_ind_hist_NPV_freepartitions':      True,
        'plot_ind_var_summary_stats':            True,
        #                                      # False,
        'plot_ind_map_topo_egid':                True,
        'plot_ind_map_node_connections':         True,
        'plot_ind_map_omitted_gwr_egids':        True,
        #                                      # False,
        'plot_agg_line_installedCap_per_month':  True,
        'plot_agg_line_productionHOY_per_node':  True,
        'plot_agg_line_gridPremiumHOY_per_node': True,
        'plot_agg_line_gridpremium_structure':   True,
        'plot_agg_line_production_per_month':    True,
        'plot_agg_line_cont_charact_new_inst':   True,
    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
# if not not dataagg_scenarios:
for k_sett, scen_sett in dataagg_scenarios.items():
    dataagg_settings = scen_sett
    data_aggregation_MASTER.data_aggregation_MASTER(dataagg_settings) if run_dataagg else print('')


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    pvalloc_initialization_MASTER.pvalloc_initialization_MASTER(pvalloc_settings) if run_alloc else print('')
    pvalloc_MCalgorithm_MASTER.pvalloc_MC_algorithm_MASTER(pvalloc_settings) if run_alloc else print('')


# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings) if run_visual else print('')



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




