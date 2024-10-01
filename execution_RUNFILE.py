import os

import data_aggregation_MASTER
import pv_allocation_MASTER
import visualization_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
# os.chdir('C:/Models/OptimalPV_RH')

months_pred =  3 #36
run_on_server = False
run_alloc = True
run_visual = False

# data_aggregation 
dataagg_scenarios = {   
    # 'preprep_BL_20to22_1and2homes_buff005':{
    #     'kt_numbers': [13,], 
    #     'year_range': [2020, 2022], 
    #     'gwr_selection_specs': {'GKLAS': ['1110','1121',],}, 
    #     'solkat_selection_specs': { 'GWR_EGID_buffer_size': 0.05,}, 
    # }, 
    # 'preprep_BL_20to22_1and2homes_buff02':{
    #     'kt_numbers': [13,],
    #     'year_range': [2020, 2022],
    #     'gwr_selection_specs': {'GKLAS': ['1110','1121',],},
    #     'solkat_selection_specs': { 'GWR_EGID_buffer_size': 0.2,},
    # },
}
dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
pvalloc_scenarios={
    f'ongoing_dev_{months_pred}m':{
        'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': [2791, 2787,],
        'create_gdf_export_of_topology':    False,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},
}

parkplatz = {
    f'smallBL_{months_pred}m_npvweight':{
        'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'create_gdf_export_of_topology':    False,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},
    f'smallBL_{months_pred}m_random':{
        'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'create_gdf_export_of_topology':    False,
        'algorithm_specs': {
            'inst_selection_method': 'random',
    }},
    #
    f'smallBL_24m_npvweight_005buff':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff005',
        'script_run_on_server': run_on_server,
        'months_prediction': 24,
        'create_gdf_export_of_topology':    False,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},
    f'smallBL_24m_TEST_SHP_Export':{
        'name_dir_import': 'preprep_BL_20to22_1and2homes_buff005',
        'script_run_on_server': run_on_server,
        'months_prediction': 24,
        'create_gdf_export_of_topology':    True,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv',
            'tweak_gridnode_df_prod_demand_fact': 100000,
    }},

    # #
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

        'plot_ind_line_productionHOY_per_node':  False,
        'plot_ind_line_installedCap_per_month':  False,
        'plot_ind_hist_NPV_freepartitions':      False,
        'plot_ind_map_topo_egid':                False,
        #                                      # False,
        'plot_agg_line_installedCap_per_month':  False,
        'plot_agg_line_productionHOY_per_node':  False,
        'plot_agg_line_gridPremiumHOY_per_node': False,
        'plot_agg_line_gridpremium_structure':   False,
        'plot_agg_line_productionHOY_per_node':  False,
        'plot_agg_line_production_per_month':    False,

        'plot_agg_line_cont_charact_new_inst':   True,

        # single plots (just show once, not for all scenarios)
        'map_ind_production': False, # NOT WORKING YET

    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
if not not dataagg_scenarios:
    for k_sett, scen_sett in dataagg_scenarios.items():
        dataagg_settings = scen_sett
        # data_aggregation_MASTER.data_aggregation_MASTER(dataagg_settings)


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    pv_allocation_MASTER.pv_allocation_MASTER(pvalloc_settings) if run_alloc else print('')
    

# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings) if run_visual else print('')



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




