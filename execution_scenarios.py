
def get_dataagg_execution_scenarios(run_on_server, scen_group_names,  ):
    scen_group_dir = {}

    all_scenarios = {
        # the large data_aggregation scenario, to preprepare split geometry data and inport slow API data
        'preprep_BLBSSO_18to23_1and2homes_API_reimport':{
            'script_run_on_server': run_on_server, 
            'kt_numbers': [13,12,11],
            'year_range': [2018, 2023], 
            'split_data_geometry_AND_slow_api': True, 
            'gwr_selection_specs': {'GKLAS': ['1110','1121','1276'],}, 
        },


        'preprep_BL_22to23_1and2homes_incl_missingEGID':{
            'script_run_on_server': run_on_server, 
            'kt_numbers': [13,], 
            'year_range': [2022, 2023],   
            'split_data_geometry_AND_slow_api': False, 
            'gwr_selection_specs': 
                {'GKLAS': ['1110','1121',],},
            'solkat_selection_specs': {
                'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
                'match_missing_EGIDs_to_solkat_TF': True, 
                'extend_dfuid_for_missing_EGIDs_to_be_unique': True,},
        },
        
        'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates':{
            'script_run_on_server': run_on_server, 
            'kt_numbers': [13,], 
            'year_range': [2022, 2023],   
            'split_data_geometry_AND_slow_api': False, 
            'gwr_selection_specs': 
                {'GKLAS': ['1110','1121',],},
            'solkat_selection_specs': {
                'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
                'match_missing_EGIDs_to_solkat_TF': True, 
                'extend_dfuid_for_missing_EGIDs_to_be_unique': False,},
        },

        'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates':{
            'script_run_on_server': run_on_server, 
            'kt_numbers': [13,11], 
            'year_range': [2022, 2023],   
            'split_data_geometry_AND_slow_api': False, 
            'gwr_selection_specs': 
                {'GKLAS': ['1110','1121',],},
            'solkat_selection_specs': {
                'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
                'match_missing_EGIDs_to_solkat_TF': True, 
                'extend_dfuid_for_missing_EGIDs_to_be_unique': False,},
        },
    }

    for scen_name in scen_group_names:
        all_scen_names = all_scenarios.keys()
        if scen_name in all_scen_names:
            scen_group_dir[scen_name] = all_scenarios[scen_name]
        else:
            print(f'Scenario <{scen_name}> not found in data aggregation scenarios') 
    
    return scen_group_dir


def get_pvalloc_execuction_scenarios(run_on_server, scen_group_names,  ):
    scen_group_dir = {}
    
    all_scenarios = {

    'pvalloc_BFS2761_1y_f2021_1mc_meth2.2_rnd_DEBUG':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2761, 
        ],
        'T0_prediction': '2021-01-01 00:00:00', 
        'months_prediction': 2,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2020],},
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}
            },  
    'pvalloc_BFS2761_12m_meth2.2_random_DEBUG':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2761, 
        ],
        'months_prediction': 2,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}
    },
    
    'pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'T0_prediction': '2013-01-01 00:00:00',
        'months_prediction': 120,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2012],},
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },

    'pvalloc_BLsml_10y_f2013_1mc_meth2.2_max':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'T0_prediction': '2013-01-01 00:00:00',
        'months_prediction': 120,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2012],},
        'algorithm_specs': {
            'inst_selection_method': 'max_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },

    'pvalloc_BLsml_10y_f2013_1mc_meth2.2_npv':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'T0_prediction': '2013-01-01 00:00:00',
        'months_prediction': 120,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2012],},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    }, 
    'pvalloc_BLsml_20y_f2003_1mc_meth2.2_npv':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'T0_prediction': '2003-01-01 00:00:00',
        'months_prediction': 240,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2002],},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },
    'pvalloc_BLsml_40y_f1983_1mc_meth2.2_npv':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'T0_prediction': '1983-01-01 00:00:00',
        'months_prediction': 480,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 1982],},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },

    'pvalloc_BLSOmed_10y_f2013_1mc_meth2.2_npv':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'T0_prediction': '2013-01-01 00:00:00',
        'months_prediction': 120,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2012],},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },
    'pvalloc_BLSOmed_20y_f2003_1mc_meth2.2_npv':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'T0_prediction': '2003-01-01 00:00:00',
        'months_prediction': 240,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 2002],},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },
    'pvalloc_BLSOmed_40y_f1983_1mc_meth2.2_npv':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'T0_prediction': '1983-01-01 00:00:00',
        'months_prediction': 480,
        'gwr_selection_specs':{
            'GBAUJ_minmax': [1920, 1982],},
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}, 
    },




    'pvalloc_BLsml_60m_10mc_meth2.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 10,}
    },
    'pvalloc_BLsml_60m_10mc_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 10,}
    },


    'pvalloc_BLsml_24m_6mc_meth2.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 6,}
    },
    'pvalloc_BLsml_24m_6mc_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 6,}
    },

    
    'pvalloc_BLsml_24m_meth2.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
    },
    'pvalloc_BLsml_24m_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
    },
    'pvalloc_BLsml_24m_meth3.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
    },
    'pvalloc_BLsml_24m_meth3.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
    },
    
    'pvalloc_BLsml_24m_meth2.2_random_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',
            'self_consumption_ifapplicable': 1, },
    },
    'pvalloc_BLsml_24m_meth2.2_npvweight_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',
            'self_consumption_ifapplicable': 1, },
    },
    'pvalloc_BLsml_24m_meth3.2_random_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',
            'self_consumption_ifapplicable': 1, },
    },
    'pvalloc_BLsml_24m_meth3.2_npvweight_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',
            'self_consumption_ifapplicable': 1, },
    },

    'pvalloc_BLsml_24m_meth2.2_random_275pr_costdcr':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',
            'estim_pvinst_cost_correctionfactor': 1.275},
    },
    'pvalloc_BLsml_24m_meth2.2_npvweight_275pr_costdcr':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',
            'estim_pvinst_cost_correctionfactor': 1.275},
    },
    'pvalloc_BLsml_24m_meth3.2_random_275pr_costdcr':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',
            'estim_pvinst_cost_correctionfactor': 1.275},
    },
    'pvalloc_BLsml_24m_meth3.2_npvweight_275pr_costdcr':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 24,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',
            'estim_pvinst_cost_correctionfactor': 1.275},
    },
    
    
    'pvalloc_BLsml_5y_meth2.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
    },
    'pvalloc_BLsml_5y_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
    },
    'pvalloc_BLsml_5y_meth3.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
    },
    'pvalloc_BLsml_5y_meth3.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
    },

    'pvalloc_BLsml_5y_meth2.2_random_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',
            'self_consumption_ifapplicable': 1, },
    },
    'pvalloc_BLsml_5y_meth2.2_npvweight_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',
            'self_consumption_ifapplicable': 1, },
    },
    'pvalloc_BLsml_5y_meth3.2_random_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',
            'self_consumption_ifapplicable': 1, },
    },
    'pvalloc_BLsml_5y_meth3.2_npvweight_SelfConsum':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 60,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',
            'self_consumption_ifapplicable': 1, },
    },

    

    'pvalloc_BLSOmed_48m_meth2.2_random':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLSOmed_48m_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },              
    'pvalloc_BLSOmed_48m_meth3.2_random':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLSOmed_48m_meth3.2_npvweight':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },


    'pvalloc_BLsml_10y_meth2.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLsml_10y_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLsml_10y_meth3.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLsml_10y_meth3.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },


    'pvalloc_BLSOmed_10y_meth2.2_random':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLSOmed_10y_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },              
    'pvalloc_BLSOmed_10y_meth3.2_random':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLSOmed_10y_meth3.2_npvweight':{
        'name_dir_import': 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
            2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
            2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
            2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
        ],
        'months_prediction': 120,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },



    }

    for scen_name in scen_group_names:
        all_scen_names = all_scenarios.keys()
        if scen_name in all_scen_names:
            scen_group_dir[scen_name] = all_scenarios[scen_name]
        else:
            print(f'Scenario <{scen_name}> not found in pv_allocation scenarios')

    return scen_group_dir