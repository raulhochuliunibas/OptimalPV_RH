
import execution_scenarios, MASTER_data_aggregation, MASTER_pvalloc_initialization, MASTER_pvalloc_MCalgorithm, MASTER_postprocess_analysis, MASTER_visualization

from auxiliary_functions import print_directory_stucture_to_txtfile
from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
run_on_server =             False
print_directory_stucture_to_txtfile(not(run_on_server))

run_dataagg =               False
run_alloc_init =            False
run_alloc_MCalg =           True

run_postprocess_analysis =  False
run_visual =                False


# data_aggregation 
dataagg_scenarios = {} # an empty dictionary, most relevant data aggregation scenario settings are stored in 
                       # in MASTER_data_aggregation.py, to keep execRunfile clean

# pv_allocation 
pvalloc_scenarios = execution_scenarios.get_pvalloc_execuction_scenarios(run_on_server,[
    'pvalloc_BFS2761_12m_meth2.2_random_DEBUG', 
    # 'pvalloc_BFS2761_1y_f2021_1mc_meth2.2_rnd_DEBUG', 
    
    # 'pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd', 
    # 'pvalloc_BLsml_10y_f2013_1mc_meth2.2_max', 
    # 'pvalloc_BLsml_10y_f2013_1mc_meth2.2_npv', 
    # 'pvalloc_BLSOmed_10y_f2013_1mc_meth2.2_npv', 
    
    # 'pvalloc_BLsml_20y_f2003_1mc_meth2.2_npv', 
    # 'pvalloc_BLSOmed_20y_f2003_1mc_meth2.2_npv', 
    
    # 'pvalloc_BLsml_40y_f1983_1mc_meth2.2_npv', 
    # 'pvalloc_BLSOmed_40y_f1983_1mc_meth2.2_npv', 
])
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)

comparison_scenarios = {
    'pvalloc_BLsml_48m_meth2.2': ['pvalloc_BLsml_48m_meth2.2_random', 'pvalloc_BLsml_48m_meth2.2_npvweight'],	
}

# postprocess_analysis
postprocessing_analysis_settings = {
    'MC_subdir_for_analysis': '*MC*1', 

    'prediction_accuracy_specs': {
        'show_plot': True,
        'show_all_scen': False,
        }

}

# vsualiastion 
visual_settings = {
    'plot_show': True,
    'remove_previous_plots': True,
    'remove_old_plot_scen_directories': True,
    'save_plot_by_scen_directory': True,
    'MC_subdir_for_plot': '*MC*1', 
    'mc_plots_individual_traces': True,     
    'node_selection_for_plots': ['8', '32', '10', '22'], # or None for all nodes

    # PLOT CHUNCK -------------------------->   [run plot,  show plot,  show all scen]
    # # for pvalloc_inital + sanitycheck
    # 'plot_ind_var_summary_stats':               [True,     False,      False], 
    # 'plot_ind_hist_pvcapaprod_sanitycheck':     [True,      True,      False], 
    # 'plot_ind_hist_radiation_rng_sanitycheck':  [True,     True,       False],
    # 'plot_ind_charac_omitted_gwr':              [True,     True,       True],
    # 'plot_ind_line_meteo_radiation':            [True,     True,       False], 
    
    # # for pvalloc_MC_algorithm 
    # 'plot_ind_line_installedCap':               [True,    True,        False],   
    # 'plot_ind_line_PVproduction':               [True,    True,        False], 
    # # bookmark => plot_ind_line_PVproduction problem? -> pvprod and feedin should be the same, no?
    # # BOOKMARK
    # 'plot_ind_line_productionHOY_per_node':     [True,    True,        False],
    # 'plot_ind_line_gridPremiumHOY_per_node':    [True,    True,        False],
    # 'plot_ind_line_gridPremium_structure':      [True,     True,        False],
    'plot_ind_hist_NPV_freepartitions':         [True,     True,       False],
    # 'plot_ind_map_topo_egid':                   [True,     True,       False],
    # 'plot_ind_map_node_connections':            [True,     True,       False],   
    # 'plot_ind_map_omitted_egids':               [True,     True,       False],
    # 'plot_ind_lineband_contcharact_newinst':    [True,    True,      False],

    # # for aggregated MC_algorithms
    # 'plot_mc_line_PVproduction':                [False,    True,       False],
    # 'plot_mc_line_gridnode_congestionHOY':      [False,    True,       False],
    # # for scenario comparison

    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
if True:
    for k_sett, scen_sett in dataagg_scenarios.items():
        dataagg_settings = scen_sett
        MASTER_data_aggregation.MASTER_data_aggregation(dataagg_settings) if run_dataagg else print('')


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    MASTER_pvalloc_initialization.MASTER_pvalloc_initialization(pvalloc_settings) if run_alloc_init else print('')
    MASTER_pvalloc_MCalgorithm.MASTER_pvalloc_MC_algorithm(pvalloc_settings) if run_alloc_MCalg else print('')


# POSTPROCESSIGN ANALYSIS RUNs ---------------------------------------------------------------
MASTER_postprocess_analysis.MASTER_postprocess_analysis(pvalloc_scenarios, postprocessing_analysis_settings) if run_postprocess_analysis else print('')



# VISUALISATION RUNs  ------------------------------------------------------------------------
MASTER_visualization.MASTER_visualization(pvalloc_scenarios, visual_settings) if run_visual else print('')



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




