
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization

pvalloc_scen_list = [
    
    # PVAllocScenario_Settings(
    #     name_dir_export                 = 'pvalloc_BLsml_test1_polars_default',
    #     name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
    #     T0_year_prediction              = 2021,
    #     months_prediction               = 36,
    #     CHECKspec_n_iterations_before_sanitycheck   = 2,
    #     ALGOspec_inst_selection_method              = 'random', 
    #     ALGOspec_rand_seed                          = 123,
    #     TECspec_pvprod_calc_method                  = 'method2.2',
    #     MCspec_montecarlo_iterations                = 2,
    #     ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test1_pandas_no_outtopo_demand',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,

        test_faster_array_computation   = False,
        GRIDspec_flat_profile_demand_dict = {'_window1':{'t': [0,24],  'demand_share': 1.0},}, 
        GRIDspec_flat_profile_demand_total_EGID = 0.0,                                            
    ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test1_partition300',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,

        ALGOspec_topo_subdf_partitioner = 300,
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test1_no_outtopo_demand',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,

        GRIDspec_flat_profile_demand_dict = {'_window1':{'t': [0,24],  'demand_share': 1.0},}, 
        GRIDspec_flat_profile_demand_total_EGID = 0.0,                                            
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test1_pvdf_adjust_TRUE',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,
        
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,
        ),

]



visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                '*test2*',
                '*test3*',
                ], 
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization = False,  
            ),    
    ]      


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        try:    
            scen_class.run_pvalloc_initalization()
        except Exception as e:
            print(f"Error during initialization: {e}")
            continue    
        try:
            scen_class.run_pvalloc_mcalgorithm()
        except Exception as e:
            print(f"Error during MC algorithm: {e}")
            continue
        # scen_class.run_pvalloc_initalization()
        # scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()


    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        # # -- def plot_ALL_init_sanitycheck(self, ): -------------
        visual_class.plot_ind_var_summary_stats()                     # runs as intended
        visual_class.plot_ind_hist_pvcapaprod_sanitycheck()           # runs as intended
        # # visual_class.plot_ind_boxp_radiation_rng_sanitycheck()
        visual_class.plot_ind_charac_omitted_gwr()                    # runs as intended
        visual_class.plot_ind_line_meteo_radiation()                  # runs as intended

        # # -- def plot_ALL_mcalgorithm(self,): -------------
        visual_class.plot_ind_line_installedCap()                     # runs as intended
        visual_class.plot_ind_line_productionHOY_per_node()           # runs as intended
        visual_class.plot_ind_line_productionHOY_per_EGID()           # runs as intended
        # visual_class.plot_ind_line_PVproduction()                     # runs
        visual_class.plot_ind_hist_NPV_freepartitions()               # runs as intended
        visual_class.plot_ind_line_gridPremiumHOY_per_node()          # runs 
        visual_class.plot_ind_line_gridPremium_structure()            # runs 




print('done')

