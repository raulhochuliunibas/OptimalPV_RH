
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization

pvalloc_scen_list = [
    
    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test3_2013_4bfs',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2013,
        months_prediction               = 120,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'prob_weighted_npv',
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 1,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test3_2013_16bfs',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [ 2767, 2771, 2761, 2762, 2769, 2764, 2765, 2773,         # BLmed with inst with / before 2008: Bottmingen, Oberwil, Aesch, Allschwil, Münchenstein, Biel-Benken, Binningen, Reinach
                                            2473, 2475, 2480, 2618, 2621, 2883, 2622, 2616,                         # SOsml: Dornach, Hochwald, Seewen, Himmelried, Nunningen, Bretzwil, Zullwil, Fehren
        ], 
        T0_year_prediction              = 2013,
        months_prediction               = 120,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'prob_weighted_npv',
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 1,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test3_2013_30y_16bfs',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [ 2767, 2771, 2761, 2762, 2769, 2764, 2765, 2773,         # BLmed with inst with / before 2008: Bottmingen, Oberwil, Aesch, Allschwil, Münchenstein, Biel-Benken, Binningen, Reinach
                                            2473, 2475, 2480, 2618, 2621, 2883, 2622, 2616,                         # SOsml: Dornach, Hochwald, Seewen, Himmelried, Nunningen, Bretzwil, Zullwil, Fehren
        ], 
        T0_year_prediction              = 2013,
        months_prediction               = 360,
        CSTRspec_iter_time_unit         = 'year',
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'prob_weighted_npv',
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 1,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,
        ),


    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test3_2013_5kt',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        kt_numbers                      = [ 2, 11, 12, 13, 26 ], 
        T0_year_prediction              = 2013,
        months_prediction               = 120,
        CSTRspec_iter_time_unit        = 'year',
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'prob_weighted_npv',
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 1,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,
        ),        
]





visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                '*test1*',
                '*test2*',
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

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
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
