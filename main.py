
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


pvalloc_scen_list = [


    PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_aggr_RUR_default_rnd_selfconsum1',
        bfs_numbers                                          = [
                                                                2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
                                                                ],          
        mini_sub_model_TF                                    = True,
        mini_sub_model_nEGIDs                                = 200,
        export_csvs                                          = True,   
        test_faster_array_computation                        = True,
        create_gdf_export_of_topology                        = True,
        TECspec_self_consumption_ifapplicable                = 1.0,
        T0_year_prediction                                   = 2021,
        months_prediction                                    = 48,
        CSTRspec_iter_time_unit                              = 'year',
        CSTRspec_ann_capacity_growth                         = 0.2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
        ALGOspec_topo_subdf_partitioner                      = 250, 
        ALGOspec_inst_selection_method                       = 'random', 
        # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
        ALGOspec_rand_seed                                   = 123,
        # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    ), 
    
    PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_aggr_RUR_southf_rnd_selfconsum1',
        bfs_numbers                                          = [
                                                                2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
                                                                ],          
        mini_sub_model_TF                                    = True,
        mini_sub_model_nEGIDs                                = 200,
        export_csvs                                          = True,   
        test_faster_array_computation                        = True,
        create_gdf_export_of_topology                        = True,
        TECspec_self_consumption_ifapplicable                = 1.0,
        T0_year_prediction                                   = 2021,
        months_prediction                                    = 48,
        CSTRspec_iter_time_unit                              = 'year',
        CSTRspec_ann_capacity_growth                         = 0.2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
        ALGOspec_topo_subdf_partitioner                      = 250, 
        ALGOspec_inst_selection_method                       = 'random', 
        # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
        ALGOspec_rand_seed                                   = 123,
        ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    ), 

    # PVAllocScenario_Settings(name_dir_export ='pvalloc_mini_aggr_URB_default_rnd_selfconsum1',
    #     bfs_numbers                                          = [
    #                                                             2773, 2769, 2770,   # URBAN: Reinach, Münchenstein, Muttenz
    #                                                             ],          
    #     mini_sub_model_TF                                    = True,
    #     mini_sub_model_nEGIDs                                = 200,
    #     test_faster_array_computation                        = True,
    #     create_gdf_export_of_topology                        = True,
    #     TECspec_self_consumption_ifapplicable                = 1.0,
    #     T0_year_prediction                                   = 2021,
    #     months_prediction                                    = 48,
    #     CSTRspec_iter_time_unit                              = 'year',
    #     CSTRspec_ann_capacity_growth                         = 0.2,
    #     ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
    #     ALGOspec_topo_subdf_partitioner                      = 250, 
    #     ALGOspec_inst_selection_method                       = 'random', 
    #     # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
    #     ALGOspec_rand_seed                                   = 123,
    #     # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    # ),  
     
    # PVAllocScenario_Settings(
    #     name_dir_export ='pvalloc_mini_aggr_RUR_max_selfconsum1',
    #     bfs_numbers                                          = [
    #                                                             2620, 2622, 2621, 2683, 2889, 2612,  # RURAL: Meltingen, Zullwil, Nunningen, Bretzwil, Lauwil, Beinwil
    #                                                             ],          
    #     mini_sub_model_TF                                    = True,
    #     mini_sub_model_nEGIDs                                = 200,
    #     test_faster_array_computation                        = True,
    #     create_gdf_export_of_topology                        = True,
    #     T0_year_prediction                                   = 2021,
    #     months_prediction                                    = 48,
    #     CSTRspec_iter_time_unit                              = 'year',
    #     CSTRspec_ann_capacity_growth                         = 0.2,
    #     ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
    #     ALGOspec_topo_subdf_partitioner                      = 250, 
    #     ALGOspec_inst_selection_method                       = 'max_npv', 
    #     # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
    #     ALGOspec_rand_seed                                   = 123,
    #     ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    # ), 

]

visualization_list = [
        Visual_Settings(
            # pvalloc_exclude_pattern_list = [
            #     '*.txt','*.xlsx','*.csv','*.parquet', '*old_vers*', ], 
            pvalloc_include_pattern_list = [
                'pvalloc_mini_aggr_*'
            ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = True,  
            remove_old_plots_in_visualization  = False,  
            remove_old_csvs_in_visualization   = False, 
    )]   

if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)
        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()


    # visualization ---------------------
    visualization_list = [
    Visual_Settings(
        pvalloc_exclude_pattern_list = [
            '*.txt','*.xlsx','*.csv','*.parquet',
            '*old_vers*',
            ], 
        pvalloc_include_pattern_list = [
            'pvalloc_mini_aggr*',
        ],
        plot_show                          = True,
        save_plot_by_scen_directory        = True, 
        remove_old_plot_scen_directories   = True,  
        remove_old_plots_in_visualization  = True,  
        remove_old_csvs_in_visualization   = False, 

        # # -- def plot_ALL_init_sanitycheck(self, ): --- [run plot,  show plot,  show all scen] ---------
        # plot_ind_var_summary_stats_TF                   = [True,      True,       False],
        # plot_ind_hist_pvcapaprod_sanitycheck_TF         = [True,      True,       False],
        # plot_ind_hist_pvprod_deviation_TF               = [True,      True,       False],
        # plot_ind_charac_omitted_gwr_TF                  = [True,      True,       False],
        # plot_ind_line_meteo_radiation_TF                = [True,      True,       False],

        # # -- def plot_ALL_mcalgorithm(self,): --------- [run plot,  show plot,  show all scen] ---------
        # # plot_ind_line_installedCap_TF                   = [True,      True,       False],
        plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False],
        plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False],
        # plot_ind_line_productionHOY_per_node_TF         = [True,      True,       False],           
        plot_ind_line_PVproduction_TF                   = [True,      True,       False],
        plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False],
        # # plot_ind_line_gridPremiumHOY_per_node_TF        = [True,      True,       False],
        # # plot_ind_line_gridPremiumHOY_per_EGID_TF        = [True,      True,       False],
        # # plot_ind_line_gridPremium_structure_TF          = [True,      True,       False],
        # # plot_ind_hist_NPV_freepartitions_TF             = [True,      True,       False],
        # # plot_ind_hist_pvcapaprod_TF                     = [True,      True,       False],
        plot_ind_map_topo_egid_TF                       = [True,      True,       False],
        # plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False],
        # plot_ind_lineband_contcharact_newinst_TF        = [True,      True,       False],
    )]

    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)
        visual_class.plot_ALL_init_sanitycheck()
        visual_class.plot_ALL_mcalgorithm()

print('done')

