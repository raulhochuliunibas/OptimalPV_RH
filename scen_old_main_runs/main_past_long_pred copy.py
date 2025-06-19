from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [

    # pvalloc_BLsml_10y_f2013_1mc_meth2.2_npv        
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_10y_f2013_1mc_meth2.2_npv',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year_prediction = 2013,            # start date for the prediction of the future construction capacity
        months_prediction  = 120,
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1,
        ),
    # pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year_prediction = 2013,            # start date for the prediction of the future construction capacity
        months_prediction  = 120,
        ALGOspec_inst_selection_method = 'random',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1,
        ),
    # pvalloc_BLsml_10y_f2013_1mc_meth2.2_max
    PVAllocScenario_Settings(
        name_dir_export    = 'pvalloc_BLsml_10y_f2013_1mc_meth2.2_max',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_year_prediction = 2013,            # start date for the prediction of the future construction capacity
        months_prediction  = 120,
        ALGOspec_inst_selection_method = 'max_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1,
        ),



    # # pvalloc_BLsml_13y_f2010_1mc_meth2.2_npv        
    # PVAllocScenario_Settings(
    #     name_dir_export    = 'pvalloc_BLsml_13y_f2010_1mc_meth2.2_npv',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     show_debug_prints  = True,
    #     bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_year_prediction = 2010,                     # start date for the prediction of the future construction capacity
    #     months_prediction  = 156,
    #     ALGOspec_inst_selection_method = 'prob_weighted_npv',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
    # # pvalloc_BLsml_13y_f2010_1mc_meth2.2_rnd
    # PVAllocScenario_Settings(
    #     name_dir_export    = 'pvalloc_BLsml_13y_f2010_1mc_meth2.2_rnd',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     show_debug_prints  = True,
    #     bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_year_prediction = 2010,                     # start date for the prediction of the future construction capacity
    #     months_prediction  = 156,
    #     ALGOspec_inst_selection_method = 'random',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
    # # pvalloc_BLsml_13y_f2010_1mc_meth2.2_max
    # PVAllocScenario_Settings(
    #     name_dir_export    = 'pvalloc_BLsml_13y_f2010_1mc_meth2.2_max',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     show_debug_prints  = True,
    #     bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_year_prediction = 2010,                     # start date for the prediction of the future construction capacity
    #     months_prediction  = 156,
    #     ALGOspec_inst_selection_method = 'max_npv',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
   

]


if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        # scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')


