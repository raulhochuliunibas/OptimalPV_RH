import os
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization

pvalloc_scen_list = [

    # SCENARIOS 2C: test2 scens with MULTIPLE HOUSING BUILDINGS + AGGRICULTURAL BUILDINGS

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test2c_default_max',
        name_dir_import                 = 'preprep_BLBSSO_22to23_extSolkatEGID_aggrfarms',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 360,
        CSTRspec_iter_time_unit         = 'year',
        GWRspec_GKLAS                               = ['1110', '1121', '1122', '1276', '1278',  ],
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations_fordev_sequentially                = 1,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,

        ALGOspec_subselec_filter_criteria = None, 
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
        pvalloc_class = PVAllocScenario(pvalloc_scen)
        
        if (pvalloc_class.sett.overwrite_scen_init) or (not os.path.exists(pvalloc_class.sett.name_dir_export_path)): 
            pvalloc_class.run_pvalloc_initalization()

        pvalloc_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        plot_method_names = [
            
        ]

        for plot_method in plot_method_names:
            try:
                method = getattr(visual_class, plot_method)
                method()
            except Exception as e:
                print(f"Error in {plot_method}: {e}")

print('\n\n -------- done -------')
