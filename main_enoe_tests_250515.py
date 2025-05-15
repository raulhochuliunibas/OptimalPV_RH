
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [
    
    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_polars_default',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints               = True,
        create_gdf_export_of_topology   = True,
        test_faster_array_computation   = True,
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year                         = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_pandas',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints               = True,
        create_gdf_export_of_topology   = True,
        test_faster_array_computation   = False,
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year                         = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_polars_no_pvdf_adjust',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints               = True,
        create_gdf_export_of_topology   = True,
        test_faster_array_computation   = True,
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year                         = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = False,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_pandas_no_pvdf_adjust',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints               = True,
        create_gdf_export_of_topology   = True,
        test_faster_array_computation   = False,
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year                         = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = False,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_polars_southfacing_max',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints               = True,
        create_gdf_export_of_topology   = True,
        test_faster_array_computation   = True,
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year                         = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        ALGOspec_subselec_filter_criteria = 'southfacing_1spec',  #'eastwest_3spec':'southfacing_1spec'
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_polars_eastwestfacing_max',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints               = True,
        create_gdf_export_of_topology   = True,
        test_faster_array_computation   = True,
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year                         = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        ALGOspec_subselec_filter_criteria = 'eastwest_3spec',  #'eastwest_3spec':'southfacing_1spec'
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2
        ),


]

if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')

