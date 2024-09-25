import os

import data_aggregation_MASTER
import pv_allocation_MASTER
import visualization_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
# os.chdir('C:/Models/OptimalPV_RH')


# data_aggregation 
dataagg_scenarios = {   
    'preprep_BL_20to22_1and2homes_buff005':{
        'kt_numbers': [13,], 
        'year_range': [2020, 2022], 
        'gwr_selection_specs': {'GKLAS': ['1110','1121',],}, 
        'solkat_selection_specs': { 'GWR_EGID_buffer_size': 0.05,}, 
    }, 
    'preprep_BL_20to22_1and2homes_buff02':{
        'kt_numbers': [13,],
        'year_range': [2020, 2022],
        'gwr_selection_specs': {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': { 'GWR_EGID_buffer_size': 0.2,},
    },
}
dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
months_pred =  12 #36
run_on_server = False
pvalloc_scenarios={
    # BL small sample, 1 y ~ca. 3h 1 scenario
    f'dev_samllBL{months_pred}m_npv': {
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff005', #'preprep_BSBLSO_21to22_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        # 'bfs_numbers': [2791, 2787,],
        'recreate_topology':                True, 
        'recalc_economics_topo_df':         True,
        'run_allocation_loop':              True,
        'create_gdf_export_of_topology':    False,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},
    f'dev_samllBL{months_pred}m_random': {
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff02', #preprep_BSBLSO_21to22_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        # 'bfs_numbers': [2791, 2787,],
        'recreate_topology':                True, 
        'recalc_economics_topo_df':         True,
        'run_allocation_loop':              True,
        'create_gdf_export_of_topology':    False,
        'algorithm_specs': {
            'inst_selection_method': 'random',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},
    f'dev_samllBL{months_pred}m_npv_TEST_GDF_EXPORT': {
        'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        # 'bfs_numbers': [2791, 2787,],
        'recreate_topology':                True, 
        'recalc_economics_topo_df':         True,
        'run_allocation_loop':              True,
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},

}

parkplatz = {
    f'dev_samllBL{months_pred}m_buff005': {
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff005',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 10000,}
            },
    f'dev_samllBL{months_pred}m_buff02': {    
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff02',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 10000, }
            }, 
    # 
    # 
    f'pvalloc_smallBL_{months_pred}m_npv_weighted': {
            'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes',
            'script_run_on_server': run_on_server,
            'months_prediction': months_pred,
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
    }},
    f'pvalloc_smallBL_{months_pred}m_random': {
            'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes',
            'script_run_on_server': run_on_server,
            'months_prediction': months_pred,
            'algorithm_specs': {
                'inst_selection_method': 'random',
    }}, 
    f'pvalloc_smallBL_{months_pred}m_npv_weighted_TEST_GDF_export': {
            'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes_w_farms',
            'script_run_on_server': run_on_server,
            'months_prediction': months_pred,
            'create_gdf_export_of_topology':    True,
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
    }}, 
    f'pvalloc_smallBL_{months_pred}m_random_TEST_GDF_export': {
            'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes_w_farms',
            'script_run_on_server': run_on_server,
            'months_prediction': months_pred,
            'create_gdf_export_of_topology':    True,
            'algorithm_specs': {
                'inst_selection_method': 'random',
    }},

}

pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': True,

        'plot_ind_line_productionHOY_per_node': True,
        'plot_ind_line_installedCap_per_month': True,
        'plot_ind_line_installedCap_per_BFS': False,
        'plot_ind_hist_NPV_freepartitions': True,
        'plot_ind_map_topo_egid': True,

        'plot_agg_line_installedCap_per_month': True,
        'plot_agg_line_productionHOY_per_node': True,
        'plot_agg_line_gridPremiumHOY_per_node': True,
        'plot_agg_line_gridpremium_structure': True,

        # single plots (just show once, not for all scenarios)
        'map_ind_production': False, # NOT WORKING YET

    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
if not not dataagg_scenarios:
    for k_sett, scen_sett in dataagg_scenarios.items():
        dataagg_settings = scen_sett
        data_aggregation_MASTER.data_aggregation_MASTER(dataagg_settings)


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    pv_allocation_MASTER.pv_allocation_MASTER(pvalloc_settings)
    

# VISUALISATION RUNs  ------------------------------------------------------------------------
# visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings)



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




