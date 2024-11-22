import data_aggregation_MASTER, pvalloc_initialization_MASTER, pvalloc_MCalgorithm_MASTER, visualization_MASTER
#mport pvalloc_postprocessing_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
months_pred = 48 #600 #36
MC_iter = 2
run_on_server = False
bfs_numbers = [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475] # Laufen & Umgebung > [2791, 2787, 2792, 2784, 2793, 2782, 2781,] # Breitenbach & Umgebung [2617, 2615, 2614, 2613, 2782, 2620, 2622]

run_dataagg =       False
run_alloc_init =    True
run_alloc_MCalg =   True
run_visual =        False


# data_aggregation 
dataagg_scenarios = {
    'preprep_BLBSSO_18to23_1and2homes_API_reimport':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,12,11],
        'year_range': [2018, 2023], 
        'split_data_geometry_AND_slow_api': True, 
        'gwr_selection_specs': {'GKLAS': ['1110','1121','1276'],}, 
    },
    'preprep_BL_22to23_1and2homes':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,], 
        # 'bfs_numbers': bfs_numbers,
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
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
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     # 'kt_numbers': [13,],
    #     'bfs_numbers': bfs_numbers,
    #     'recreate_topology':             True, 
    #     'recalc_economics_topo_df':      True,
    #     'sanitycheck_byEGID':            True,
    #     'create_gdf_export_of_topology': True,

    #     # 'sanitycheck_summary_byEGID_specs':{
    #     #     'n_iterations_before_sanitycheck': 2,},
    #     'algorithm_specs': {
    #         'inst_selection_method': 'prob_weighted_npv',},
    #     'tech_economic_specs': {
    #         'max_distance_m_for_EGID_node_matching': 0,
    #         'self_consumption_ifapplicable': 0,
    #         'pvprod_calc_method': 'method2',},
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': MC_iter,},
    # },
    f'pvalloc_{months_pred}m_meth2_npv':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': bfs_numbers,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2',},
        'MC_loop_specs': {
            'montecarlo_iterations': MC_iter,},
    },
        f'pvalloc_{months_pred}m_meth3_npv':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes',
        'script_run_on_server': run_on_server,
        'months_prediction': months_pred,
        'bfs_numbers': bfs_numbers,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3',},
        'MC_loop_specs': {
            'montecarlo_iterations': MC_iter,},
    },
}
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': False,
        'MC_subdir_for_plot': 'ASDFASDF', 
        'node_selection_for_plots': ['node1', 'node10', 'node15'], # or None for all nodes

        # PLOT IND for pvalloc_initalization + sanity check
        'plot_ind_var_summary_stats':            True,        # |> bookmark: add a histplot for pvprod_kwh.sum and STROMERTRAG.sum for comparison
        'plot_ind_charac_omitted_gwr':           False,       # |> bookmark: if possible find a way to continue script, even if plots are shown
        #NEU plot radiation over time
        #NEU 'plot_ind_hist_installedCap_kw': > in plot_ind_var_summary_stats? 
        #NEU plot_ind_hist_annualpvprod_kwh : > in plot_ind_var_summary_stats?


        # PLOT IND for pvalloc_MC_algorithm 

        'plot_ind_line_installedCap':            False,       # |> bookmark: Necessary to specify which MC iteration to plot! ctrlF: "MC_subdir_for_plot"
        'plot_ind_line_productionHOY_per_node':  False,      
        # 'plot_ind_line_installedCap_per_month':  False,     # |> bookmark: remove 
        'plot_ind_hist_NPV_freepartitions':      False,


        # for aggregated pvalloc_MC_algorithm 

        # PLOT IND for maps

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




