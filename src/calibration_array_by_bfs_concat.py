from calibration_class import Calibration_Settings, Calibration

if __name__ == '__main__':
    preprep_list = [
        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs3',
            name_calib_subscen='allCHbfs',
            kt_numbers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,], 
            # bfs_numbers=[5391, 5394, 
            #              120, 4726, 4761, 4651, 
            #              4203, 4204,
            #              ],
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
                        'rfr_mod_name': '_rfr1a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None, 'full_dist')],
                        }, 
                    'mod2b': {
                        'rfr_mod_name': '_rfr2b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None, 'full_dist')],
                       }, 

                    'mod3a': {
                        'rfr_mod_name': '_rfr3a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist', ), 
                            ( 5.0,  7.5,    'segment_dist', ),
                            ( 7.5,  10.0,   'segment_dist', ),
                            ( 10.0, 12.5,   'segment_dist', ),
                            ( 12.5, 17.5,   'segment_dist', ),
                            ( 17.5, 25.0,   'segment_dist', ),
                            ( 25.0, 100.0,  'segment_dist', ),
                        ], 
                    }, 
                    'mod3b': {
                        'rfr_mod_name': '_rfr3b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist'), 
                            ( 5.0,  7.5,    'segment_dist'), 
                            ( 7.5,  10.0,   'full_dist'), 
                            ( 10.0, 12.5,   'full_dist'), 
                            ( 12.5, 17.5,   'full_dist'), 
                            ( 17.5, 25.0,   'full_dist'), 
                            ( 25.0, 100.0,  'full_dist'), 
                        ],
                    }, 
            }},
        ),

        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs3',
            name_calib_subscen='BS12BL13bfs',
            kt_numbers=[12, 13, ],
            # bfs_numbers=[5391, 5394, 
            #              120, 4726, 4761, 4651, 
            #              4203, 4204,
            #              ],
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
                        'rfr_mod_name': '_rfr1a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None, 'full_dist')],
                        }, 
                    'mod2b': {
                        'rfr_mod_name': '_rfr2b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None, 'full_dist')],
                       }, 

                    'mod3a': {
                        'rfr_mod_name': '_rfr3a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist', ), 
                            ( 5.0,  7.5,    'segment_dist', ),
                            ( 7.5,  10.0,   'segment_dist', ),
                            ( 10.0, 12.5,   'segment_dist', ),
                            ( 12.5, 17.5,   'segment_dist', ),
                            ( 17.5, 25.0,   'segment_dist', ),
                            ( 25.0, 100.0,  'segment_dist', ),
                        ], 
                    }, 
                    'mod3b': {
                        'rfr_mod_name': '_rfr3b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist'), 
                            ( 5.0,  7.5,    'segment_dist'), 
                            ( 7.5,  10.0,   'full_dist'), 
                            ( 10.0, 12.5,   'full_dist'), 
                            ( 12.5, 17.5,   'full_dist'), 
                            ( 17.5, 25.0,   'full_dist'), 
                            ( 25.0, 100.0,  'full_dist'), 
                        ],
                    }, 
            }},
        ), 

        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs3',
            name_calib_subscen='BE2bfs',
            kt_numbers=[2,], 
            # bfs_numbers=[5391, 5394, 
            #              120, 4726, 4761, 4651, 
            #              4203, 4204,
            #              ],
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
                        'rfr_mod_name': '_rfr1a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None, 'full_dist')],
                        }, 
                    'mod2b': {
                        'rfr_mod_name': '_rfr2b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None, 'full_dist')],
                       }, 

                    'mod3a': {
                        'rfr_mod_name': '_rfr3a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist', ), 
                            ( 5.0,  7.5,    'segment_dist', ),
                            ( 7.5,  10.0,   'segment_dist', ),
                            ( 10.0, 12.5,   'segment_dist', ),
                            ( 12.5, 17.5,   'segment_dist', ),
                            ( 17.5, 25.0,   'segment_dist', ),
                            ( 25.0, 100.0,  'segment_dist', ),
                        ], 
                    }, 
                    'mod3b': {
                        'rfr_mod_name': '_rfr3b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist'), 
                            ( 5.0,  7.5,    'segment_dist'), 
                            ( 7.5,  10.0,   'full_dist'), 
                            ( 10.0, 12.5,   'full_dist'), 
                            ( 12.5, 17.5,   'full_dist'), 
                            ( 17.5, 25.0,   'full_dist'), 
                            ( 25.0, 100.0,  'full_dist'), 
                        ],
                    }, 
            }},
        ), 

        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs3',
            name_calib_subscen='AG192bfs',
            kt_numbers=[19,], 
            # bfs_numbers=[5391, 5394, 
            #              120, 4726, 4761, 4651, 
            #              4203, 4204,
            #              ],
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
                        'rfr_mod_name': '_rfr1a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None, 'full_dist')],
                        }, 
                    'mod2b': {
                        'rfr_mod_name': '_rfr2b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None, 'full_dist')],
                       }, 

                    'mod3a': {
                        'rfr_mod_name': '_rfr3a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist', ), 
                            ( 5.0,  7.5,    'segment_dist', ),
                            ( 7.5,  10.0,   'segment_dist', ),
                            ( 10.0, 12.5,   'segment_dist', ),
                            ( 12.5, 17.5,   'segment_dist', ),
                            ( 17.5, 25.0,   'segment_dist', ),
                            ( 25.0, 100.0,  'segment_dist', ),
                        ], 
                    }, 
                    'mod3b': {
                        'rfr_mod_name': '_rfr3b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.0,  5.0,    'segment_dist'), 
                            ( 5.0,  7.5,    'segment_dist'), 
                            ( 7.5,  10.0,   'full_dist'), 
                            ( 10.0, 12.5,   'full_dist'), 
                            ( 12.5, 17.5,   'full_dist'), 
                            ( 17.5, 25.0,   'full_dist'), 
                            ( 25.0, 100.0,  'full_dist'), 
                        ],
                    }, 
            }},
        ), 
    ]
            
    for sett in preprep_list:
        calib_class = Calibration(sett)
        # calib_class.concatenate_prerep_data()           if calib_class.sett.run_concatenate_preprep_data_TF else None

        calib_class.approach2_regression_instsize()     if calib_class.sett.run_approach2_regression_instsize_TF else None
        calib_class.random_forest_regression()          if calib_class.sett.run_appr2_random_forest_reg_TF else None

        # calib_class.approach1_fit_optim_cost_function()     if calib_class.sett.run_approach2_regression_instsize_TF else None





"""
        Calibration_Settings(
            name_dir_export='calib_all_CH_bfs3',
            name_calib_subscen='reg2_all_CH_bfs',
            kt_numbers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,], 
            # bfs_numbers=[5391, 5394, 
            #              120, 4726, 4761, 4651, 
            #              4203, 4204,
            #              ],
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
                        'rfr_mod_name': '_rfr1_highfit', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2    ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None)],

                }, 
                    'mod1b': {
                        'rfr_mod_name': '_rfr1_lowfit', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         150  ,    # default: 100   # | 1,       
                        'min_samples_split':    50    ,    # default: 2     # | 1000,    
                        'max_depth':            10   ,    # default: None  # | 3,       
                        'kWp_segments': [(None, None)],
                }, 


                    'mod2b': {
                        'rfr_mod_name': '_rfr2b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None)],
                       }, 
                    'mod2c': {
                        'rfr_mod_name': '_rfr2c', 
                        'df_suffix': '_pvroof20to80',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None)],
                       }, 
                    # 'mod2d': {
                    #     'rfr_mod_name': '_rfr2d', 
                    #     'df_suffix': '_res1to2',

                    #     'random_state':         None,    # default: None  # | None,    
                    #     'n_jobs':               -1,      # default: None  # | -1,  
                    #     'cross_validation':     None, 
                    #     'n_estimators':         600  ,    # default: 100   # | 1,       
                    #     'min_samples_split':    2     ,    # default: 2     # | 1000,    
                    #     'max_depth':            40   ,    # default: None  # | 3,
                    #     'kWp_segments': [(None, None)],
                    #    }, 
                    'mod2e': {
                        'rfr_mod_name': '_rfr2e',
                        'df_suffix': '_pvroof20to60',
                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1
                        'cross_validation':     None,
                        'n_estimators':         600  ,    # default: 100   # | 1,
                        'min_samples_split':    2     ,    # default: 2     # | 1000,
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None)],
                        },
                    'mod2f': {
                        'rfr_mod_name': '_rfr2f',
                        'df_suffix': '_pvroof20to50',
                        'random_state':         None,    # default: None  # | None,
                        'n_jobs':               -1,      # default: None  # | -1
                        'cross_validation':     None,
                        'n_estimators':         600  ,    # default: 100   # | 1,
                        'min_samples_split':    2     ,    # default: 2     # | 1000,
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [(None, None)],
                        },

                    'mod3a': {
                        'rfr_mod_name': '_rfr3a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
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
                    'mod3b': {
                        'rfr_mod_name': '_rfr3b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
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
                #     'mod3c': {
                #         'rfr_mod_name': '_rfr3c', 
                #         'df_suffix': '_pvroof20to80',

                #         'random_state':         None,    # default: None  # | None,    
                #         'n_jobs':               -1,      # default: None  # | -1,  
                #         'cross_validation':     None, 
                #         'n_estimators':         600  ,    # default: 100   # | 1,       
                #         'min_samples_split':    2     ,    # default: 2     # | 1000,    
                #         'max_depth':            40   ,    # default: None  # | 3,
                #         'kWp_segments': [
                #             (0, 5), 
                #             (5, 7.5),
                #             (7.5, 10),
                #             (10, 12.5), 
                #             (12.5, 17.5),
                #             (17.5, 25),
                #             (25, 100), 
                #         ], 
                # }, 


                'mod4a': {
                        'rfr_mod_name': '_rfr4a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
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
                # 'mod4b': {
                #         'rfr_mod_name': '_rfr4b', 
                #         'df_suffix': '_pvroof20to70',

                #         'random_state':         None,    # default: None  # | None,    
                #         'n_jobs':               -1,      # default: None  # | -1,  
                #         'cross_validation':     None, 
                #         'n_estimators':         600  ,    # default: 100   # | 1,       
                #         'min_samples_split':    2     ,    # default: 2     # | 1000,    
                #         'max_depth':            40   ,    # default: None  # | 3,
                #         'kWp_segments': [
                #             ( 0.00,  6.00), 
                #             ( 6.00,  8.00),
                #             ( 8.00,  9.00),
                #             ( 9.00, 10.00),
                #             (10.00, 11.00), 
                #             (11.00, 12.00),
                #             (12.00, 14.00),
                #             (14.00, 16.00),
                #             (16.00, 20.00),
                #             (20.00, 99.00)
                #         ], 
                # }, 
                # 'mod4c': {
                #         'rfr_mod_name': '_rfr4c', 
                #         'df_suffix': '_pvroof20to80',

                #         'random_state':         None,    # default: None  # | None,    
                #         'n_jobs':               -1,      # default: None  # | -1,  
                #         'cross_validation':     None, 
                #         'n_estimators':         600  ,    # default: 100   # | 1,       
                #         'min_samples_split':    2     ,    # default: 2     # | 1000,    
                #         'max_depth':            40   ,    # default: None  # | 3,
                #         'kWp_segments': [
                #             ( 0.00,  6.00), 
                #             ( 6.00,  8.00),
                #             ( 8.00,  9.00),
                #             ( 9.00, 10.00),
                #             (10.00, 11.00), 
                #             (11.00, 12.00),
                #             (12.00, 14.00),
                #             (14.00, 16.00),
                #             (16.00, 20.00),
                #             (20.00, 99.00)
                #         ], 
                # },
                'mod5a': {
                        'rfr_mod_name': '_rfr5a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.00,  3.00),
                            ( 3.00,  4.00),
                            ( 4.00,  5.00),
                            ( 5.00,  6.00),
                            ( 6.00,  7.00),
                            ( 7.00,  8.00),
                            ( 8.00,  9.00),
                            ( 9.00, 10.00),
                            (10.00, 11.00),
                            (11.00, 12.00),
                            (12.00, 13.00),
                            (13.00, 14.00),
                            (14.00, 16.00),
                            (16.00, 19.00),
                            (19.00, 23.00),
                            (23.00, 28.00),
                            (28.00, 99.00)                        ], 
                },  
                'mod5b': {
                        'rfr_mod_name': '_rfr5b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'kWp_segments': [
                            ( 0.00,  3.00),
                            ( 3.00,  4.00),
                            ( 4.00,  5.00),
                            ( 5.00,  6.00),
                            ( 6.00,  7.00),
                            ( 7.00,  8.00),
                            ( 8.00,  9.00),
                            ( 9.00, 10.00),
                            (10.00, 11.00),
                            (11.00, 12.00),
                            (12.00, 13.00),
                            (13.00, 14.00),
                            (14.00, 16.00),
                            (16.00, 19.00),
                            (19.00, 23.00),
                            (23.00, 28.00),
                            (28.00, 99.00)                        ], 
                },  


                'mod6a': {
                        'rfr_mod_name': '_rfr6a', 
                        'df_suffix': '',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'stdev_factor': 0.5,  
                        'kWp_segments': [
                            ( 0.00,  3.00),
                            ( 3.00,  4.00),
                            ( 4.00,  5.00),
                            ( 5.00,  6.00),
                            ( 6.00,  7.00),
                            ( 7.00,  8.00),
                            ( 8.00,  9.00),
                            ( 9.00, 10.00),
                            (10.00, 11.00),
                            (11.00, 12.00),
                            (12.00, 13.00),
                            (13.00, 14.00),
                            (14.00, 16.00),
                            (16.00, 19.00),
                            (19.00, 23.00),
                            (23.00, 28.00),
                            (28.00, 99.00)
                        ], 
                },  
                'mod6b': {
                        'rfr_mod_name': '_rfr6b', 
                        'df_suffix': '_pvroof20to70',

                        'random_state':         None,    # default: None  # | None,    
                        'n_jobs':               -1,      # default: None  # | -1,  
                        'cross_validation':     None, 
                        'n_estimators':         600  ,    # default: 100   # | 1,       
                        'min_samples_split':    2     ,    # default: 2     # | 1000,    
                        'max_depth':            40   ,    # default: None  # | 3,
                        'stdev_factor': 0.5,  
                        'kWp_segments': [
                            ( 0.00,  3.00),
                            ( 3.00,  4.00),
                            ( 4.00,  5.00),
                            ( 5.00,  6.00),
                            ( 6.00,  7.00),
                            ( 7.00,  8.00),
                            ( 8.00,  9.00),
                            ( 9.00, 10.00),
                            (10.00, 11.00),
                            (11.00, 12.00),
                            (12.00, 13.00),
                            (13.00, 14.00),
                            (14.00, 16.00),
                            (16.00, 19.00),
                            (19.00, 23.00),
                            (23.00, 28.00),
                            (28.00, 99.00)
                            ], 
                }, 
                
                
                'mod7a': {
                    'rfr_mod_name': '_rfr7a', 
                    'df_suffix': '',

                    'random_state':         None,    # default: None  # | None,    
                    'n_jobs':               -1,      # default: None  # | -1,  
                    'cross_validation':     None, 
                    'n_estimators':         600  ,    # default: 100   # | 1,       
                    'min_samples_split':    2     ,    # default: 2     # | 1000,    
                    'max_depth':            40   ,    # default: None  # | 3,
                    'kWp_segments': [
                        ( 0,  5),
                        ( 5, 10),
                        (10, 15),
                        (15, 25),
                        (25, 99),
                        ], 
                },  
            }},
        ),
"""