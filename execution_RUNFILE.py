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
months_pred = 5 #600 #36
MC_iter = 10
run_on_server = False
bfs_numbers = [2791, 2787, 2792, 2784, 2793, 2782, 2781,]

run_dataagg =       True
run_alloc_init =    True
run_alloc_MCalg =   True
run_visual =        True


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
        'kt_numbers': [13,], 
        'bfs_numbers': bfs_numbers,
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
        'demand_specs':{
            'input_data_source': "NETFLEX",},
        'gwr_selection_specs': 
            {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': {
            'test_loop_optim_buff_size': False, },
    }, 
}

dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
pvalloc_scenarios={
    # f'pvalloc_DEV_{months_pred}m_meth2_npv':{
    'pvalloc_DEV_5m_meth2_selfcon00_DirDiffRad':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': bfs_numbers,
        
        'recreate_topology':             True, 
        'recalc_economics_topo_df':      True,
        'sanitycheck_byEGID':            True,
        'create_gdf_export_of_topology': True,

        'sanitycheck_summary_byEGID_specs':{
            'n_iterations_before_sanitycheck': 1,},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',},
        'tech_economic_specs': {
            'self_consumption_ifapplicable': 0,},
        'MC_loop_specs': {
            'montecarlo_iterations': MC_iter,},
        'weather_specs': {
            'meteo_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
        },

    # f'pvalloc_DEV_{months_pred}m_meth2_rand':{
    #     'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     'bfs_numbers': bfs_numbers,
        # 'algorithm_specs': {
            # 'inst_selection_method': 'random',},
    #     'tech_economic_specs': {
    #         'self_consumption_ifapplicable': 0,},
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': MC_iter,},
    #     'weather_specs': {
    #         ' meteo_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
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
            'meteo_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',]}
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
            'meteo_col_radiation_proxy': ['Basel Direct Shortwave Radiation',]}
        },  
}
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': False,
        'MC_subdir_for_plot': 'ASDFASDF', 
        'node_selection_for_plots': ['node1', 'node10', 'node15'], # or None for all nodes

        # for pvalloc_initalization + sanity check
        'plot_ind_var_summary_stats':            True,
        'plot_ind_charac_omitted_gwr':           False,


        # for pvalloc_MC_algorithm 

        'plot_ind_line_productionHOY_per_node':  False,
        'plot_ind_line_installedCap_per_month':  False,
        'plot_ind_hist_NPV_freepartitions':      False,

        # for aggregated pvalloc_MC_algorithm 


        #                                      # False,
        'plot_ind_map_topo_egid':                False,
        'plot_ind_map_node_connections':         False,
        'plot_ind_map_omitted_gwr_egids':        False,
        #                                      # False,
        'plot_agg_line_installedCap_per_month':  False,
        'plot_agg_line_productionHOY_per_node':  False,
        'plot_agg_line_gridPremiumHOY_per_node': False,
        'plot_agg_line_gridpremium_structure':   False,
        'plot_agg_line_production_per_month':    False,
        'plot_agg_line_cont_charact_new_inst':   False,
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
    pvalloc_initialization_MASTER.pvalloc_initialization_MASTER(pvalloc_settings) if run_alloc_init else print('')
    pvalloc_MCalgorithm_MASTER.pvalloc_MC_algorithm_MASTER(pvalloc_settings) if run_alloc_MCalg else print('')


# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings) if run_visual else print('')



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




