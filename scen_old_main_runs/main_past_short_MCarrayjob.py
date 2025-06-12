
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [

    # pvalloc_BLsml_3y_f2020_4mc_npv
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_3y_f2020_1mc_npv.part400',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        
        T0_year_prediction = 2020,            
        months_prediction  = 36, 
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        ALGOspec_topo_subdf_partitioner = 99999999, 
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 4
        ),

    # MC ITERATIONS IN SEQUENCE AND AS ARRAY JOB!

]


if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')

