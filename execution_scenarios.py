
def get_pvalloc_execuction_scenarios(run_on_server, scen_group_names,  ):
    scen_group_dir = {}
    
    all_scenarios = {

    'pvalloc_BFS2761_12m_meth2.2_random_DEBUG':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2761, 
        ],
        'months_prediction': 12,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}
    },
    'pvalloc_BFS2761_12m_meth3.2_random_DEBUG':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2761, 
        ],
        'months_prediction': 12,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 1,}
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




    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],        
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind_DFUID_duplicates':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID_DF_UID_duplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],        
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind_DFUID_duplicates':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID_DF_UID_duplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    




    # old - archived

    'pvalloc_BLsml_48m_meth2.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLsml_48m_meth2.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method2.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLsml_48m_meth3.2_random':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'random', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    'pvalloc_BLsml_48m_meth3.2_npvweight':{
        'name_dir_import': 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'months_prediction': 48,
        'algorithm_specs': {
            'inst_selection_method': 'prob_weighted_npv', },
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'MC_loop_specs': {
            'montecarlo_iterations': 2,}
    },
    

    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    
    'pvalloc_BLsml_1roof_12m_meth2.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },  

    'pvalloc_BLsml_07roof_12m_meth2.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },



    'pvalloc_BLSOsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLSOsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLSOsml_07roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLSOsml_07roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'gwr_selection_specs': {
            'solkat_area_per_EGID_range': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLSOsml_1roof_12m_meth2.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLSOsml_1roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLSOsml_07roof_12m_meth2.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLSOsml_07roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785,
            2473, 2475, 2480,   
        ],
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    }

    for scen_name in scen_group_names:
        all_scen_names = all_scenarios.keys()
        if scen_name in all_scen_names:
            scen_group_dir[scen_name] = all_scenarios[scen_name]

    return scen_group_dir