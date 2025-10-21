from calibration_class import Calibration_Settings, Calibration

if __name__ == '__main__':
    preprep_list = [
        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs2',
            name_calib_subscen='reg2_all_CH_bfs',
            # kt_numbers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,], 
            bfs_numbers=[5391, 5394, 
                         120, 4726, 4761, 4651, 
                         4203, 4204,
                         ],
            concat_bfs_subsample              = True, 

            n_rows_import                     = None,
            rerun_import_and_preprp_data_TF   = True,
            export_gwr_ALL_building_gdf_TF    = False, 

            run_concatenate_preprep_data_TF         = True,
            run_approach1_fit_optim_costfunction_TF = True, 
            run_approach2_regression_instsize_TF    = True,
            run_appr2_random_forest_reg_TF          = True,

            reg2_random_forest_reg_settings         = {
                'run_ML_rfr_TF': True,
                'reg2_rfrname_dfsuffix_dicts': {
                    'mod1': {
                        'rfr_mod_name': '_rfr1', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         [100, ]  ,    # default: 100   # | 1,       
                        'min_samples_split':    [20, ]    ,    # default: 2     # | 1000,    
                        'max_depth':            [10, ]   ,    # default: None  # | 3,       
                }, 
                #     'mod2': {
                #         'rfr_mod_name': '_rfr2',
                #         'df_suffix': '_kwpmax20',
                #         'random_state':         None,    # default: None  # | None,
                #         'n_jobs':               -1,      # default: None  # | -1,
                #         'cross_validation':     None,
                #         'n_estimators':         [1, ]  ,    # default: 100   # | 1,
                #         'min_samples_split':    [5, ]    ,    # default: 2     # | 1000,
                #         'max_depth':            [3, ]   ,    # default: None  # | 3,
                # },
                    'mod3': {
                        'rfr_mod_name': '_rfr1', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         [200, ]  ,    # default: 100   # | 1,       
                        'min_samples_split':    [2 ]    ,    # default: 2     # | 1000,    
                        'max_depth':            [20, ]   ,    # default: None  # | 3,       
                }, 

            }}

        ), 
    ]
            
    for sett in preprep_list:
        calib_class = Calibration(sett)
        calib_class.concatenate_prerep_data()           if calib_class.sett.run_concatenate_preprep_data_TF else None

        calib_class.approach2_regression_instsize()     if calib_class.sett.run_approach2_regression_instsize_TF else None
        # calib_class.random_forest_regression()          if calib_class.sett.run_appr2_random_forest_reg_TF else None


