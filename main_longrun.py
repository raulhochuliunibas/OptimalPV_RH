
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


# pv alloctaion ------------------------------------------
pvalloc_scen_list = [

    PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_Longrun_aggr_RUR_max',
        bfs_numbers                                          = [
                                                    # 2767, 2771, 2775, 2764,                               # SEMI-URBAN: Bottmingen, Oberwil, Therwil, Biel-Benken
                                                    # 2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
                                                    2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren
                                                                ],          
        mini_sub_model_TF                                    = True,
        mini_sub_model_by_X                                  = 'by_EGID',
        mini_sub_model_nEGIDs                                = 9900,
        create_gdf_export_of_topology                        = True,
        export_csvs                                          = True,
        T0_year_prediction                                   = 2021,
        months_prediction                                    = 120, 
        TECspec_self_consumption_ifapplicable                = 0.8,
        CSTRspec_iter_time_unit                              = 'year',
        CSTRspec_ann_capacity_growth                         = 0.2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
        ALGOspec_topo_subdf_partitioner                      = 250, 
        ALGOspec_pvinst_size_calculation                     = 'npv_optimized',   # 'inst_full_partition' / 'npv_optimized'
        ALGOspec_inst_selection_method                       = 'max_npv', 
        # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
        ALGOspec_rand_seed                                   = 123,
        # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    ), 

    # PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_Longrun_aggr_RUR_max_origSFMFHdemand',
    #     bfs_numbers                                          = [
    #                                                 # 2767, 2771, 2775, 2764,                               # SEMI-URBAN: Bottmingen, Oberwil, Therwil, Biel-Benken
    #                                                 # 2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
    #                                                 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren
    #                                                             ],          
    #     mini_sub_model_TF                                    = True,
    #     mini_sub_model_nEGIDs                                = 600,
    #     test_faster_array_computation                        = True,
    #     create_gdf_export_of_topology                        = True,
    #     T0_year_prediction                                   = 2021,
    #     months_prediction                                    = 240, 
    #     TECspec_self_consumption_ifapplicable                = 1.0,
    #     CSTRspec_iter_time_unit                              = 'year',
    #     CSTRspec_ann_capacity_growth                         = 0.2,
    #     ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
    #     ALGOspec_tweak_demand_profile                        = 1,
    #     ALGOspec_topo_subdf_partitioner                      = 250, 
    #     ALGOspec_inst_selection_method                       = 'max_npv', 
    #     # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
    #     ALGOspec_rand_seed                                   = 123,
    #     # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    # ), 

    # PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_Longrun_aggr_RUR_default_rnd',
    #     bfs_numbers                                          = [
    #                                                 # 2767, 2771, 2775, 2764,                               # SEMI-URBAN: Bottmingen, Oberwil, Therwil, Biel-Benken
    #                                                 # 2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
    #                                                 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren
    #                                                             ],          
    #     mini_sub_model_TF                                    = True,
    #     mini_sub_model_nEGIDs                                = 600,
    #     # export_csvs                                          = True,   
    #     test_faster_array_computation                        = True,
    #     create_gdf_export_of_topology                        = True,
    #     T0_year_prediction                                   = 2021,
    #     months_prediction                                    = 240, 
    #     TECspec_self_consumption_ifapplicable                = 1.0,
    #     # GWRspec_solkat_max_n_partitions                      = 3,
    #     CSTRspec_iter_time_unit                              = 'year',
    #     CSTRspec_ann_capacity_growth                         = 0.2,
    #     ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
    #     ALGOspec_topo_subdf_partitioner                      = 250, 
    #     ALGOspec_inst_selection_method                       = 'random', 
    #     # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
    #     ALGOspec_rand_seed                                   = 123,
    #     # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    # ), 

    # PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_Longrun_aggr_RUR_southf_rnd',
    #     bfs_numbers                                          = [
    #                                                 # 2767, 2771, 2775, 2764,                               # SEMI-URBAN: Bottmingen, Oberwil, Therwil, Biel-Benken
    #                                                 # 2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
    #                                                 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren
    #                                                             ],          
    #     mini_sub_model_TF                                    = True,
    #     mini_sub_model_nEGIDs                                = 600,
    #     test_faster_array_computation                        = True,
    #     create_gdf_export_of_topology                        = True,
    #     T0_year_prediction                                   = 2021,
    #     months_prediction                                    = 240, 
    #     TECspec_self_consumption_ifapplicable                = 1.0,
    #     CSTRspec_iter_time_unit                              = 'year',
    #     CSTRspec_ann_capacity_growth                         = 0.2,
    #     ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
    #     ALGOspec_topo_subdf_partitioner                      = 250, 
    #     ALGOspec_inst_selection_method                       = 'max_npv', 
    #     # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
    #     ALGOspec_rand_seed                                   = 123,
    #     ALGOspec_subselec_filter_criteria = 'southfacing_1spec', 
    # ), 



]


# visualization ------------------------------------------
visualization_list = [
    Visual_Settings(
        pvalloc_exclude_pattern_list = [
            '*.txt','*.xlsx','*.csv','*.parquet',
            '*old_vers*',
            ], 
        pvalloc_include_pattern_list = [
            # 'pvalloc_mini_Longrun_aggr_RUR_default_rnd_selfconsum1',
            # 'pvalloc_mini_Longrun_aggr_RUR_southf_rnd_selfconsum1', 
            # 'pvalloc_mini_Longrun_aggr_RUR*', 
            'pvalloc_mini_aggr_RUR*', 
        ],
        plot_show                          = False,
        save_plot_by_scen_directory        = True, 
        remove_old_plot_scen_directories   = False,  
        remove_old_plots_in_visualization  = False,  
        remove_old_csvs_in_visualization   = False, 
        cut_timeseries_to_zoom_hour        = True,

        default_zoom_hour                  = [212*24, (212*24)+(24*14)], 

        # # # -- def plot_ALL_init_sanitycheck(self, ): --- [run plot,  show plot,  show all scen] ---------
        # plot_ind_var_summary_stats_TF                   = [True,      True,       False],
        # # plot_ind_hist_pvcapaprod_sanitycheck_TF         = [True,      True,       False],
        # # plot_ind_hist_pvprod_deviation_TF               = [True,      True,       False],
        # plot_ind_charac_omitted_gwr_TF                  = [True,      True,       False],
        # # plot_ind_line_meteo_radiation_TF                = [True,      True,       False],

        # # # # -- def plot_ALL_mcalgorithm(self,): --------- [run plot,  show plot,  show all scen] ---------
        # # # plot_ind_line_installedCap_TF                   = [True,      True,       False],
        # plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False],                   
        # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False],                   
        # # plot_ind_line_productionHOY_per_node_TF         = [True,      True,       False],                   
        # # plot_ind_line_PVproduction_TF                   = [True,      True,       False],                   
        # plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False],                   
        # # # plot_ind_line_gridPremiumHOY_per_node_TF        = [True,      True,       False],
        # # # plot_ind_line_gridPremiumHOY_per_EGID_TF        = [True,      True,       False],
        # # # plot_ind_line_gridPremium_structure_TF          = [True,      True,       False],
        # # # plot_ind_hist_NPV_freepartitions_TF             = [True,      True,       False],
        # # # plot_ind_hist_pvcapaprod_TF                     = [True,      True,       False],
        # plot_ind_map_topo_egid_TF                       = [True,      True,       False],
        # plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False],
        # # # plot_ind_lineband_contcharact_newinst_TF        = [True,      True,       False],

        plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False],                   
        plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False],                   
        plot_ind_line_productionHOY_per_node_TF         = [True,      True,       False],                   
        plot_ind_line_PVproduction_TF                   = [True,      True,       False],                   
        plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False],                   
        plot_ind_map_topo_egid_TF                       = [True,      True,       False],
        plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False],

        plot_ind_mapline_prodHOY_EGIDrfcombo_specs = {
                'specific_selected_EGIDs': [
                    # '190145779', 
                    # '190178022', 
                    # '190296588', 
                                            ], 
                'rndm_sample_seed': 123,
                'n_rndm_egid_winst_pvdf': 2, 
                'n_rndm_egid_winst_alloc': 2,
                'n_rndm_egid_woinst': 0,
                'n_rndm_egid_outsample': 0, 
                'n_partition_pegid_minmax': (1,2), 
                'roofpartition_color': '#fff2ae',
                'actual_ts_trace_marker': 'cross'
            },
    )
    ]


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)
        # scen_class.run_pvalloc_initalization()
        # scen_class.run_pvalloc_mcalgorithm()

    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)
        visual_class.plot_ALL_init_sanitycheck()
        visual_class.plot_ALL_mcalgorithm()

print('done')

