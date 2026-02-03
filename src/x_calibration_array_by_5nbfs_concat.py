from calibration_class import Calibration_Settings, Calibration

if __name__ == '__main__':
    preprep_list = [
        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs_TEST',
            name_calib_subscen='reg2_all_CH_bfs',
            # kt_numbers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,], 
            bfs_numbers=[
                4203, 4204, 4726, 4761, 5391, 4201, 4206, 4207, 4210, 4200, 4002, 4008, 4001, 4003, 4012, 
                ],
            # concat_bfs_subsample              = True, 

            n_rows_import                     = None,
            rerun_import_and_preprp_data_TF   = True,
            export_gwr_ALL_building_gdf_TF    = False, 

            run_concatenate_preprep_data_TF         = True,
            run_approach1_fit_optim_costfunction_TF = True, 
            run_approach2_regression_instsize_TF    = True,
            run_appr2_random_forest_reg_TF          = True,

            reg2_random_forest_reg_settings         = {
                'run_ML_rfr_TF': True,
                'visualize_ML_rfr_TF': True,
                'reg2_rfrname_dfsuffix_dicts': {
                # #     'mod_GRIDSEARCH_CONVOLUTION_2': {
                # #         'rfr_mod_name': '_rfr2',
                # #         'df_suffix': '_kwpmax20',
                # #         'random_state':         None,    # default: None  # | None,
                # #         'n_jobs':               -1,      # default: None  # | -1,
                # #         'cross_validation':     None,
                # #         'n_estimators':         [1, ]  ,    # default: 100   # | 1,
                # #         'min_samples_split':    [5, ]    ,    # default: 2     # | 1000,
                # #         'max_depth':            [3, ]   ,    # default: None  # | 3,
                # # },
                    'mod1a': {
                        'rfr_mod_name': '_rfr1', 
                        'df_suffix': '',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         400  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None)],

                }, 
                    'mod1b': {
                        'rfr_mod_name': '_rfr1', 
                        'df_suffix': '',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         150  ,    # default: 100   # | 1,       
                        'min_samples_split':    50    ,    # default: 2     # | 1000,    
                        'max_depth':            10   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None)],

                }, 
                    'mod2': {
                        'rfr_mod_name': '_rfr2', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         400  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            30   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None)],
       
                }, 
                    'mod3': {
                        'rfr_mod_name': '_rfr3', 
                        'df_suffix': '_res1to2',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         400  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None)],
                }, 
                    'mod4': {
                        'rfr_mod_name': '_rfr4', 
                        'df_suffix': '',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         20  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            30   ,    # default: None  # | 3,       
                        'kWp_segments': [
                            (0, 5), 
                            (5, 7.5),
                            (7.5, 10),
                            (10, 12.5), 
                            (12.5, 17.5),
                            (17.5, 25),
                            (25, 100), 
                        ], 
                }, 
                    'mod5': {
                        'rfr_mod_name': '_rfr5', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         20  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            30   ,    # default: None  # | 3,       
                        'kWp_segments': [
                            (0, 5), 
                            (5, 7.5),
                            (7.5, 10),
                            (10, 12.5), 
                            (12.5, 17.5),
                            (17.5, 25),
                            (25, 100), 
                        ], 
                }, 
                'mod6': {
                        'rfr_mod_name': '_rfr6', 
                        'df_suffix': '',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         20  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            30   ,    # default: None  # | 3,       
                        'kWp_segments': [
                            ( 0.00,  6.00), 
                            ( 6.00,  8.00),
                            ( 8.00,  9.00),
                            ( 9.00, 10.00),
                            (10.00, 11.00), 
                            (11.00, 12.00),
                            (12.00, 14.00),
                            (14.00, 16.00),
                            (16.00, 20.00),
                            (20.00, 99.00)
                        ], 
                }, 
                    'mod7': {
                        'rfr_mod_name': '_rfr7', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         24,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         20  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            30   ,    # default: None  # | 3,       
                        'kWp_segments': [
                            ( 0.00,  6.00), 
                            ( 6.00,  8.00),
                            ( 8.00,  9.00),
                            ( 9.00, 10.00),
                            (10.00, 11.00), 
                            (11.00, 12.00),
                            (12.00, 14.00),
                            (14.00, 16.00),
                            (16.00, 20.00),
                            (20.00, 99.00)
                        ], 
                }, 


            }},

            opt1_kWp_optimization_subs_settings = {
                'no_subs': {
                    'opt_suffix': '_s0-00', 
                    'inst_subsidy': 0.0,
                },
                '20_percent': {
                    'opt_suffix': '_s0-20', 
                    'inst_subsidy': 0.2,
                },
                '30_percent': {
                    'opt_suffix': '_s0-30', 
                    'inst_subsidy': 0.3,
                },
            },



        ),
    ]
            
    for sett in preprep_list:
        calib_class = Calibration(sett)
        # calib_class.concatenate_prerep_data()           if calib_class.sett.run_concatenate_preprep_data_TF else None

        calib_class.approach2_regression_instsize()     if calib_class.sett.run_approach2_regression_instsize_TF else None
        calib_class.random_forest_regression()          if calib_class.sett.run_appr2_random_forest_reg_TF else None

        # calib_class.approach1_fit_optim_cost_function()     if calib_class.sett.run_approach2_regression_instsize_TF else None



