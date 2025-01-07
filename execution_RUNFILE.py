
import execution_scenarios, data_aggregation_MASTER, pvalloc_initialization_MASTER, pvalloc_MCalgorithm_MASTER, visualization_MASTER
#mport pvalloc_postprocessing_MASTER

from auxiliary_functions import print_directory_stucture_to_txtfile
from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
run_on_server =     False
print_directory_stucture_to_txtfile(not(run_on_server))

run_dataagg =       False
run_alloc_init =    False
run_alloc_MCalg =   False
run_visual =        True


# data_aggregation 
dataagg_scenarios = {
    # 'preprep_BLBSSO_18to23_1and2homes_API_reimport':{
    #     'script_run_on_server': run_on_server, 
    #     'kt_numbers': [13,12,11],
    #     'year_range': [2018, 2023], 
    #     'split_data_geometry_AND_slow_api': True, 
    #     'gwr_selection_specs': {'GKLAS': ['1110','1121','1276'],}, 
    # },
    'preprep_BL_22to23_1and2homes_incl_missingEGID':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,], 
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
        'gwr_selection_specs': 
            {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': {
            'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
            'match_missing_EGIDs_to_solkat_TF': True, 
            'extend_dfuid_for_missing_EGIDs_to_be_unique': True,},
    },
    
    'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,], 
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
        'gwr_selection_specs': 
            {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': {
            'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
            'match_missing_EGIDs_to_solkat_TF': True, 
            'extend_dfuid_for_missing_EGIDs_to_be_unique': False,},
    },

    'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,11], 
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
        'gwr_selection_specs': 
            {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': {
            'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
            'match_missing_EGIDs_to_solkat_TF': True, 
            'extend_dfuid_for_missing_EGIDs_to_be_unique': False,},
    },

}
dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
pvalloc_scenarios = execution_scenarios.get_pvalloc_execuction_scenarios(run_on_server,[
    # 'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind', 
    # 'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind', 
    # 'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind_DFUID_duplicates', 
    # 'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind_DFUID_duplicates',

    'pvalloc_BLsml_48m_meth2.2_random',
    'pvalloc_BLsml_48m_meth2.2_npvweight',
    'pvalloc_BLsml_48m_meth3.2_random',
    'pvalloc_BLsml_48m_meth3.2_npvweight',

    # 'pvalloc_BLsml_24m_meth2.2_random',
    # 'pvalloc_BLsml_24m_meth2.2_npvweight',
    # 'pvalloc_BLsml_24m_meth3.2_random',
    # 'pvalloc_BLsml_24m_meth3.2_npvweight',
    
    # 'pvalloc_BLSOmed_48m_meth2.2_random',
    # 'pvalloc_BLSOmed_48m_meth2.2_npvweight',
    # 'pvalloc_BLSOmed_48m_meth3.2_random',
    # 'pvalloc_BLSOmed_48m_meth3.2_npvweight',

    # 'pvalloc_BLsml_10y_meth2.2_random',
    # 'pvalloc_BLsml_10y_meth2.2_npvweight',
    # 'pvalloc_BLsml_10y_meth3.2_random',
    # 'pvalloc_BLsml_10y_meth3.2_npvweight',

    # 'pvalloc_BLSOmed_10y_meth2.2_random', 
    # 'pvalloc_BLSOmed_10y_meth2.2_npvweight',
    # 'pvalloc_BLSOmed_10y_meth3.2_random',
    # 'pvalloc_BLSOmed_10y_meth3.2_npvweight',


])
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)

comparison_scenarios = {
    'pvalloc_BLsml_48m_meth2.2': ['pvalloc_BLsml_48m_meth2.2_random', 'pvalloc_BLsml_48m_meth2.2_npvweight'],	
}

# vsualiastion 
visual_settings = {
        'plot_show': True,
        'remove_previous_plots': True,
        'remove_old_plot_scen_directories': True,
        'save_plot_by_scen_directory': True,
        'MC_subdir_for_plot': '*MC*1', 
        'node_selection_for_plots': ['8', '32', '10', '22'], # or None for all nodes

        # PLOT CHUNCK -------------------------> [run plot,  show plot,  show all scen]
        # for pvalloc_inital + sanitycheck
        'plot_ind_var_summary_stats':            [False,     False,      False], 
        'plot_ind_hist_pvcapaprod_sanitycheck':  [False,      True,      False], 
            'plot_ind_hist_pvcapaprod_sanitycheck_specs': {
                'uniform_scencolor_and_KDE_TF': True,
                'export_spatial_data_for_prod0': True, 
            },
        'plot_ind_charac_omitted_gwr':           [False,     True,       True],
        'plot_ind_line_meteo_radiation':         [False,     True,       False], 
        
        # for pvalloc_MC_algorithm 
        'plot_ind_line_installedCap':            [False,    True,        False],   
        'plot_ind_line_PVproduction':            [False,    True,        False], 
        # bookmark => plot_ind_line_PVproduction problem? -> pvprod and feedin should be the same, no?
        'plot_ind_line_productionHOY_per_node':  [False,    True,        False],
        'plot_ind_line_gridPremiumHOY_per_node': [False,    True,        False],
        'plot_ind_hist_NPV_freepartitions':      [False,     True,       False],
        'plot_ind_map_topo_egid':                [False,     True,       False],
        'plot_ind_map_node_connections':         [False,     True,       False],   
        'plot_ind_map_omitted_egids':            [False,     True,       False],

        'plot_ind_lineband_contcharact_newinst': [True,    True,      False],

        # for aggregated MC_algorithms

        # for scenario comparison

    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
# if not not dataagg_scenarios:/
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




