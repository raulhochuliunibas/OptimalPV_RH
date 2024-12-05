import data_aggregation_MASTER, pvalloc_initialization_MASTER, pvalloc_MCalgorithm_MASTER, visualization_MASTER
#mport pvalloc_postprocessing_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
months_pred = 1 #600 #36
MC_iter = 1
run_on_server = False
bfs_numbers = [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475] # Breitenbach & Umgebung [2617, 2615, 2614, 2613, 2782, 2620, 2622]

run_dataagg =       False
run_alloc_init =    True
run_alloc_MCalg =   True
run_visual =        True


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
    # f'pvalloc_DEV_{months_pred}m_meth2_rnd':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'months_prediction': months_pred,
    #     # 'kt_numbers': [13,],
    #     'bfs_numbers': bfs_numbers,
    #     'recreate_topology':             True, 
    #     'recalc_economics_topo_df':      True,
    #     'sanitycheck_byEGID':            True,
    #     'create_gdf_export_of_topology': True,

    #     'sanitycheck_summary_byEGID_specs':{
    #         'n_iterations_before_sanitycheck': 2,},
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method2',},
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': MC_iter,},
    # }, 
    
    'pvallco_BL_small_12m_meth3_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3',},
        'weather_specs': {
            'radiation_to_pvprod_method': 'flat',
            'flat_direct_rad_factor': 1,
            'flat_diffuse_rad_factor': 0.2,},
    },
    'pvallco_BL_small_12m_meth3_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3',},
        'weather_specs': {
            'radiation_to_pvprod_method': 'dfuid_ind',
            'flat_direct_rad_factor': 1,
            'flat_diffuse_rad_factor': 0.2,},
    },
    'pvallco_BL_small_12m_meth2_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2',},
        'weather_specs': {
            'radiation_to_pvprod_method': 'flat',
            'flat_direct_rad_factor': 1,
            'flat_diffuse_rad_factor': 0.2,}
    },
    'pvallco_BL_small_12m_meth2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2',},
        'weather_specs': {
            'radiation_to_pvprod_method': 'dfuid_ind',
            'flat_direct_rad_factor': 1,
            'flat_diffuse_rad_factor': 0.2,},
    },

    # 'pvalloc_BLsml_12m_1mc_meth2_panel1506_dir1_diff0_flatWK':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'bfs_numbers': [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475],
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'months_prediction': 12,
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': 1,},

    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method2',
    #         'kWpeak_per_m2': 0.15,    
    #         'share_roof_area_available': 0.6},
    #     'weather_specs': {
    #         'flat_direct_rad_factor': 1,
    #         'flat_diffuse_rad_factor': 0,},
    #     'panel_efficiency_specs': {
    #         'variable_panel_efficiency_TF': False,},
    # },
    # 'pvalloc_BLsml_12m_1mc_meth2_panel2010_dir1_diff0_flatWK':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'bfs_numbers': [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475],
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'months_prediction': 12,
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': 1,},

    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method2',
    #         'kWpeak_per_m2': 0.2,    
    #         'share_roof_area_available': 1},
    #     'weather_specs': {
    #         'flat_direct_rad_factor': 1,
    #         'flat_diffuse_rad_factor': 0,},
    #     'panel_efficiency_specs': {
    #         'variable_panel_efficiency_TF': False,},
    # },
    # 'pvalloc_BLsml_12m_1mc_meth2_panel1506_dir10_diff1_flatWK':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'bfs_numbers': [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475],
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'months_prediction': 12,
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': 1,},

    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method2',
    #         'kWpeak_per_m2': 0.15,    
    #         'share_roof_area_available': 0.6},
    #     'weather_specs': {
    #         'flat_direct_rad_factor': 10,
    #         'flat_diffuse_rad_factor': 1,},
    #     'panel_efficiency_specs': {
    #         'variable_panel_efficiency_TF': False,},
    # },
    
    # 'pvalloc_BLsml_12m_1mc_meth3_panel1506_dir1_diff0_flatWK':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'bfs_numbers': [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475],
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'months_prediction': 12,
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': 1,},

    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method3',
    #         'kWpeak_per_m2': 0.15,    
    #         'share_roof_area_available': 0.6},
    #     'weather_specs': {
    #         'flat_direct_rad_factor': 1,
    #         'flat_diffuse_rad_factor': 0,},
    #     'panel_efficiency_specs': {
    #         'variable_panel_efficiency_TF': False,},
    # },
    # 'pvalloc_BLsml_12m_1mc_meth3_panel2010_dir1_diff0_flatWK':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'bfs_numbers': [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475],
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'months_prediction': 12,
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': 1,},

    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method3',
    #         'kWpeak_per_m2': 0.2,    
    #         'share_roof_area_available': 1},
    #     'weather_specs': {
    #         'flat_direct_rad_factor': 1,
    #         'flat_diffuse_rad_factor': 0,},
    #     'panel_efficiency_specs': {
    #         'variable_panel_efficiency_TF': False,},
    # },
    # 'pvalloc_BLsml_12m_1mc_meth3_panel1506_dir10_diff1_flatWK':{
    #     'name_dir_import': 'preprep_BL_22to23_1and2homes',
    #     'script_run_on_server': run_on_server,
    #     'bfs_numbers': [2768, 2761, 2772, 2473, 2475, 2785, 2480, 2475],
    #     'algorithm_specs': {
    #         'inst_selection_method': 'random',},
    #     'months_prediction': 12,
    #     'MC_loop_specs': {
    #         'montecarlo_iterations': 1,},

    #     'tech_economic_specs': {
    #         'pvprod_calc_method': 'method3',
    #         'kWpeak_per_m2': 0.15,    
    #         'share_roof_area_available': 0.6},
    #     'weather_specs': {
    #         'flat_direct_rad_factor': 10,
    #         'flat_diffuse_rad_factor': 1,},
    #     'panel_efficiency_specs': {
    #         'variable_panel_efficiency_TF': False,},
    # },
    
}



parkplat = {
}
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': True,
        'remove_previous_plots': True,
        'remove_old_plot_scen_directories': True,
        'save_plot_by_scen_directory': True,
        'MC_subdir_for_plot': '*MC*1', 
        'node_selection_for_plots': ['8', '32', '10', '22'], # or None for all nodes

        # PLOT CHUCK -------------------------> [run plot,  show plot]
        # for pvalloc_inital + sanitycheck
        'plot_ind_var_summary_stats':            [False,    False], 
        'plot_ind_hist_pvcapaprod_sanitycheck':  [True,     True], 
        'plot_ind_charac_omitted_gwr':      [False,         True],  # |> bookmark: NOT WORKING properly!! how to make such that code continues if plot still shown?
        'plot_ind_line_meteo_radiation':         [True,     False], 
        # for pvalloc_MC_algorithm 
        'plot_ind_line_installedCap':            [False,    True],        
        'plot_ind_line_productionHOY_per_node':  [False,     True],  
        'plot_ind_hist_NPV_freepartitions':      [False,    False], 
        #NEU 'plot_ind_hist_installedCap_kw': > in plot_ind_var_summary_stats?
        # |CURRENT WORKING ON > hist für FLACHE*70%*kWpeak_m2 for all egids
        # |done> hist für FLACHE*70%*kWpeak_m2 only for buildings in pvdf => good comparison!
        'plot_ind_hist_pvcapaprod':              [False,     True],
        'plot_ind_map_topo_egid':                [True,    True],
        'plot_ind_map_node_connections':         [True,    True],   
        # for scen aggregation

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




