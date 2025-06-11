
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization

pvalloc_scen_list = [
    
    PVAllocScenario_Settings(
            name_dir_export    = 'pvalloc_mini_2m_2mc_rnd',
            name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
            show_debug_prints                                    = True,
            export_csvs                                          = True,
            mini_sub_model_TF                                    = True,
            mini_sub_model_nEGIDs                                = 100, 
            create_gdf_export_of_topology                        = False, 
            test_faster_array_computation                        = True,
            T0_year_prediction                                   = 2021,
            months_prediction                                    = 60,
            CSTRspec_iter_time_unit                              = 'year',
            CSTRspec_ann_capacity_growth                         = 0.2,
            CHECKspec_n_iterations_before_sanitycheck            = 2,
            ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
            ALGOspec_topo_subdf_partitioner                      = 250, 
            ALGOspec_inst_selection_method                       = 'random', 
            # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
            ALGOspec_rand_seed                                   = 123,
            # CSTRspec_constr_capa_overshoot_fact                  = 0.75,
            ALGOspec_subselec_filter_criteria                    = None,
            ALGOspec_subselec_filter_area_perc_first             = 0.5,
            TECspec_pvprod_calc_method                           = 'method2.2',
            MCspec_montecarlo_iterations                         = 1,
    ),

]


visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                ], 
            pvalloc_include_pattern_list = [
                '*test2*', 
            ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization  = False,  
            remove_old_csvs_in_visualization   = False, 
            ),    
    ]       


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        # scen_class.run_pvalloc_mcalgorithm()
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

            # # # -- def plot_ALL_mcalgorithm(self,): -------------
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
            # # "plot_ind_map_node_connections"                   # status not noted
        ]

        for plot_method in plot_method_names:
            try:
                method = getattr(visual_class, plot_method)
                method()
            except Exception as e:
                print(f"Error in {plot_method}: {e}")

print('\n\n -------- done -------')
