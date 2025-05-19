
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [
    
    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test2_default_w_pvdf_adjust',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 120,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,

        ALGOspec_subselec_filter_criteria = None, 
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test2_southfacing_max',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 120,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,

        ALGOspec_subselec_filter_criteria = 'southfacing_1spec', 
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test2_eastwestfacing_max',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 120,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,

        ALGOspec_subselec_filter_criteria = 'eastwestfacing_3spec', 
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_test2_southwestfacing_max',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 120,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'max_npv', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,

        ALGOspec_subselec_filter_criteria = 'southwestfacing_3spec', 
        ),        
]

if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')

