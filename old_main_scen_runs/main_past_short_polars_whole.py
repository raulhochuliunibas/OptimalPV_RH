
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [

    # pvalloc_BLsml_3m_f2020_1mc_npv.part9999_polars
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_3m_f2020_1mc_npv.part9999_polars',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [
            2768, 2761, 2772, 2785, 2787,
            2473, 2475, 2480,        
        ],
        T0_year_prediction = 2020,            
        months_prediction  = 3, 
        test_faster_array_computation = True,
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        ALGOspec_topo_subdf_partitioner = 99999999, 
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1
        ),

]


if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')

