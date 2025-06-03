
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


pvalloc_scen_list = [
    # PVAllocScenario_Settings(
    #     name_dir_export='pvalloc_mini_BYMONTH_rnd',
    #     mini_sub_model_TF= True,
    #     test_faster_array_computation= True,
    #     create_gdf_export_of_topology = True,
    #     T0_year_prediction                                   = 2021,
    #     months_prediction                                    = 30,
    #     CSTRspec_iter_time_unit                              = 'month',
    #     ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
    #     ALGOspec_topo_subdf_partitioner                      = 250, 
    #     ALGOspec_inst_selection_method                       = 'random', 
    #     # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
    #     ALGOspec_rand_seed                                   = 123,
    # ), 
    PVAllocScenario_Settings(
        name_dir_export ='pvalloc_mini_BYYEAR_rnd',
        mini_sub_model_TF                                    = True,
        mini_sub_model_nEGIDs                                = 50,
        test_faster_array_computation                        = True,
        create_gdf_export_of_topology                        = True,
        T0_year_prediction                                   = 2021,
        months_prediction                                    = 60,
        CSTRspec_iter_time_unit                              = 'year',
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
        ALGOspec_topo_subdf_partitioner                      = 250, 
        ALGOspec_inst_selection_method                       = 'random', 
        # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
        ALGOspec_rand_seed                                   = 123,
    ),    

]

visualization_list = [

        Visual_Settings(
            # pvalloc_exclude_pattern_list = [
            #     '*.txt','*old_vers*', 
            #     'pvalloc_BLsml_10y_f2013_1mc_meth2.2_npv'
            #     ], 
            pvalloc_include_pattern_list = [
                'pvalloc_mini_BYMONTH_rnd',
                'pvalloc_mini_BYYEAR_rnd',
                ],
            save_plot_by_scen_directory        = False, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization = False,  
            ),    
    ]    


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()


    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        plot_method_names = [
            
            # # -- def plot_ALL_init_sanitycheck(self, ): -------------
            # visual_class.plot_ind_var_summary_stats()                     # runs as intended
            # visual_class.plot_ind_hist_pvcapaprod_sanitycheck()           # runs as intended
            # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
            # visual_class.plot_ind_charac_omitted_gwr()                    # runs as intended
            # visual_class.plot_ind_line_meteo_radiation()                  # runs as intended

            # # -- def plot_ALL_mcalgorithm(self,): -------------
            # "plot_ind_line_installedCap",                     # runs as intended
            # "plot_ind_line_productionHOY_per_node",           # runs as intended
            # "plot_ind_line_productionHOY_per_EGID",           # runs as intended
            # # "plot_ind_line_PVproduction",                   # runs — optional, uncomment if needed
            # "plot_ind_hist_NPV_freepartitions",               # runs as intended
            # "plot_ind_line_gridPremiumHOY_per_node",          # runs
            # "plot_ind_line_gridPremium_structure",            # runs
            # "plot_ind_lineband_contcharact_newinst",          # status not noted
            # "plot_ind_map_topo_egid",                         # runs as intended
            # "plot_ind_map_topo_egid_incl_gridarea",         # runs as intended — optional
            # "plot_ind_map_node_connections"                   # status not noted
        ]

        for plot_method in plot_method_names:
            try:
                method = getattr(visual_class, plot_method)
                method()
            except Exception as e:
                print(f"Error in {plot_method}: {e}")

print('done')

