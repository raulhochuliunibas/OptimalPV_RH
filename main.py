
from code.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario

pvalloc_scen_list = [
    # PVAllocScenario_Settings(
    #         name_dir_export    = 'pvalloc_BFS2761_2m_f2021_1mc_meth2.2_rnd_DEBUG',
    #         name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #         show_debug_prints  = True,
    #         export_csvs        = True,
    #         T0_prediction      = '2021-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #         months_prediction  = 2,
    #         GWRspec_GBAUJ_minmax = [1920, 2020],
    #         ALGOspec_inst_selection_method = 'random',
    #         TECspec_pvprod_calc_method = 'method2.2',
    #         MCspec_montecarlo_iterations = 2,
    # ), 
    # pvalloc_BLsml_3y_f2020_10mc_meth2.2_npv        
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_3y_f2020_10mc_meth2.2_npv',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2019-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 36, 
        GWRspec_GBAUJ_minmax = [1920, 2019],
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 10
        ),
    # pvalloc_BLsml_3y_f2020_10mc_meth2.2_rnd
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_3y_f2020_10mc_meth2.2_rnd',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2019-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 36, 
        GWRspec_GBAUJ_minmax = [1920, 2019],
        ALGOspec_inst_selection_method = 'random',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 10
        ),
    # pvalloc_BLsml_3y_f2020_10mc_meth2.2_max
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_3y_f2020_10mc_meth2.2_max',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2019-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 36, 
        GWRspec_GBAUJ_minmax = [1920, 2019],
        ALGOspec_inst_selection_method = 'max_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 10
        ),



    # pvalloc_BLsml_13y_f2010_5mc_meth2.2_npv        
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_5mc_meth2.2_npv',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    # pvalloc_BLsml_13y_f2010_5mc_meth2.2_rnd
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_5mc_meth2.2_rnd',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'random',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    # pvalloc_BLsml_13y_f2010_5mc_meth2.2_max
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_5mc_meth2.2_max',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'max_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),



]

if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

print('done')

