import os

import data_aggregation_MASTER
import pv_allocation_MASTER
import visualization_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
months_pred =  2 #36
run_on_server = False

run_dataagg =   False
run_alloc =     True
run_visual =    True


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
        'bfs_numbers': [2791, 2787, 2792, 2784, 2793, 2782, 2781,],
        'year_range': [2020, 2022], 
        'split_data_geometry_AND_slow_api': False, 
        'gwr_selection_specs': {'GKLAS': ['1110','1121',],}, 
    }, 
}
dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
pvalloc_scenarios={
    f'pvalloc_DEV_{months_pred}m':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': [2791, 2787,],
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},

}
parklplatz = {
    f'pvalloc_smallBL_{months_pred}m_npv_alloc':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': [2791, 2787, 2792, 2784, 2793, 2782, 2781,],
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 1,
    }},
    f'pvalloc_smallBL_{months_pred}m_rand_alloc':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff002',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': [2791, 2787, 2792, 2784, 2793, 2782, 2781,],
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'random',
            'tweak_gridnode_df_prod_demand_fact': 1,
    }},

}
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': False,
        'node_selection_for_plots': ['node1', 'node10', 'node15'], # or None for all nodes

        'plot_ind_line_productionHOY_per_node':  False,
        'plot_ind_line_installedCap_per_month':  False,
        'plot_ind_hist_NPV_freepartitions':      False,
        'plot_ind_var_summary_stats':            False,
        #                                      # False,
        'plot_ind_map_topo_egid':                True,
        'plot_ind_map_node_connections':         False,
        'plot_ind_map_omitted_gwr_egids':        True,
        #                                      # False,
        'plot_agg_line_installedCap_per_month':  True,
        'plot_agg_line_productionHOY_per_node':  True,
        'plot_agg_line_gridPremiumHOY_per_node': True,
        'plot_agg_line_gridpremium_structure':   False,
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
    pv_allocation_MASTER.pv_allocation_MASTER(pvalloc_settings) if run_alloc else print('')
    

# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings) if run_visual else print('')



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




