
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [

    # pvalloc_BLsml_3y_f2020_1mc_npv.part400
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_3y_f2020_1mc_npv.part400',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2019-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 36, 
        GWRspec_GBAUJ_minmax = [1920, 2019],
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        ALGOspec_topo_subdf_partitioner = 400, 
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1
        ),
    # pvalloc_BLsml_3y_f2020_1mc_meth2.2_npv

    # pvalloc_BLsml_3y_f2020_1mc_npv.part9999
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_3y_f2020_1mc_npv.part9999',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2019-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 36, 
        GWRspec_GBAUJ_minmax = [1920, 2019],
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

