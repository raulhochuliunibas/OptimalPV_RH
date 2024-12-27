
def get_pvalloc_execuction_scenarios(run_on_server, scen_group_names,  ):
    scen_group_dir = {}
    

    all_scenarios = {
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],        
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1000,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BLSO_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'bfs_numbers': [
            2768, 2761, 2772, 2785, 
        ],
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1000,},
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
            'solkat_max_area_per_EGID': 1000,},
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
            'solkat_max_area_per_EGID': 1000,},
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
            'solkat_max_area_per_EGID': 1000,},
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
            'solkat_max_area_per_EGID': 1000,},
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