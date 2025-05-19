
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [
    
    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_polars_default',
        name_dir_import                 = 'preprep_BL_22to23_extSolkatEGID',
        bfs_numbers                     = [2767, 2771, 2765, 2764,  ], 
        T0_year_prediction              = 2021,
        months_prediction               = 36,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations                = 2,
        ),

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_BLsml_pandas_no_outtopo_demand',
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
        name_dir_export                 = 'pvalloc_BLsml_partition300',
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
        name_dir_export                 = 'pvalloc_BLsml_no_outtopo_demand',
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
        name_dir_export                 = 'pvalloc_BLsml_pvdf_adjust_TRUE',
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

if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')

