
import data_aggregation_MASTER, pvalloc_initialization_MASTER, pvalloc_MCalgorithm_MASTER, visualization_MASTER
#mport pvalloc_postprocessing_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
months_pred = 1 #600 #36
MC_iter = 1
run_on_server = True
bfs_numbers = [2768, 2761, 2772, 2473, 2475, 2785, 2480,] # Breitenbach & Umgebung [2617, 2615, 2614, 2613, 2782, 2620, 2622]

run_dataagg =       True
run_alloc_init =    True
run_alloc_MCalg =   True
run_visual =        True


# data_aggregation 
dataagg_scenarios = {
    # 'preprep_BLBSSO_18to23_1and2homes_API_reimport':{
    #     'script_run_on_server': run_on_server, 
    #     'kt_numbers': [13,12,11],
    #     'year_range': [2018, 2023], 
    #     'split_data_geometry_AND_slow_api': True, 
    #     'gwr_selection_specs': {'GKLAS': ['1110','1121','1276'],}, 
    # },
    'preprep_BL_22to23_1and2homes':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,], 
        # 'bfs_numbers': bfs_numbers,
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
        'gwr_selection_specs': 
            {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': {
            'match_missing_EGIDs_to_solkat_TF': False, },
    }, 
    'preprep_BL_22to23_1and2homes_incl_missingEGID':{
        'script_run_on_server': run_on_server, 
        'kt_numbers': [13,], 
        # 'bfs_numbers': bfs_numbers,
        'year_range': [2022, 2023],   
        'split_data_geometry_AND_slow_api': False, 
        'gwr_selection_specs': 
            {'GKLAS': ['1110','1121',],},
        'solkat_selection_specs': {
            'cols_adjust_for_missEGIDs_to_solkat': ['FLAECHE','STROMERTRAG'],
            'match_missing_EGIDs_to_solkat_TF': True, },
    }, 
}
dataagg_scenarios = extend_dataag_scen_with_defaults(dataagg_scenarios)


# pv_allocation 
pvalloc_scenarios = {

    
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
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
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
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
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    
    'pvalloc_BLsml_1roof_12m_meth2.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
   
    'pvalloc_BLsml_07roof_12m_meth2.2_rad_dfuid_ind':{
                'script_run_on_server': run_on_server,
                'tech_economic_specs': {
                    'share_roof_area_available': 0.7,
                    'pvprod_calc_method': 'method2.2',},
                'weather_specs': {
                    'rad_rel_loc_max_by': 'dfuid_specific',
                    'radiation_to_pvprod_method': 'dfuid_ind',}
            },
    'pvalloc_BLsml_07roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
 
}

parkplatz = {
    'pvalloc_BLsml_07roof_12m_meth2.1_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 0.7, 
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_12m_meth2.1_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_12m_meth2.2_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_12m_meth2.2_rad_dfuid_ind':{
                'script_run_on_server': run_on_server,
                'tech_economic_specs': {
                    'share_roof_area_available': 0.7,
                    'pvprod_calc_method': 'method2.2',},
                'weather_specs': {
                    'rad_rel_loc_max_by': 'dfuid_specific',
                    'radiation_to_pvprod_method': 'dfuid_ind',}
            },

    'pvalloc_BLsml_07roof_12m_meth3.1_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_12m_meth3.1_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_12m_meth3.2_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.1_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},                             
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.1_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.2_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth3.1_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth3.1_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth3.2_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_07roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 0.7,
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    

    
    'pvalloc_BLsml_1roof_12m_meth2.1_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_12m_meth2.1_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_12m_meth2.2_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_12m_meth2.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLsml_1roof_12m_meth3.1_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_12m_meth3.1_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_12m_meth3.2_rad_flat':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_12m_meth3.2_rad_dfuid_ind':{
        'script_run_on_server': run_on_server,
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.1_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},                             
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.1_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method2.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.1_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.1_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.1',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_flat':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'flat',}
    },
    'pvalloc_BLsml_1roof_extSolkatEGID_12m_meth3.2_rad_dfuid_ind':{
        'name_dir_import': 'preprep_BL_22to23_1and2homes_incl_missingEGID',
        'script_run_on_server': run_on_server,
        'gwr_selection_specs': {
            'solkat_max_area_per_EGID': 1500,},
        'tech_economic_specs': {
            'share_roof_area_available': 1, 
            'pvprod_calc_method': 'method3.2',},
        'weather_specs': {
            'rad_rel_loc_max_by': 'dfuid_specific',
            'radiation_to_pvprod_method': 'dfuid_ind',}
    },

}
pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': True,
        'remove_previous_plots': True,
        'remove_old_plot_scen_directories': True,
        'save_plot_by_scen_directory': True,
        'MC_subdir_for_plot': '*MC*1', 
        'node_selection_for_plots': ['8', '32', '10', '22'], # or None for all nodes

        # PLOT CHUNCK -------------------------> [run plot,  show plot,  show all scen]
        # for pvalloc_inital + sanitycheck
        'plot_ind_var_summary_stats':            [False,     True,        True], 
        'plot_ind_hist_pvcapaprod_sanitycheck':  [True,      True,       True], 
        'plot_ind_charac_omitted_gwr':           [False,      True,        True],
        'plot_ind_line_meteo_radiation':         [False,     True,      False], 
        # for pvalloc_MC_algorithm 
        'plot_ind_line_installedCap':            [False,    True,      False],       
        'plot_ind_line_productionHOY_per_node':  [False,    True,      False],
        'plot_ind_hist_NPV_freepartitions':      [False,    False,     False],
        'plot_ind_hist_pvcapaprod':              [False,     True],  # |> bookmark

        'plot_ind_map_topo_egid':                [False,     True,       True],
        'plot_ind_map_node_connections':         [False,     True,       False],   
        
        # still to be updated
        'plot_ind_map_omitted_gwr_egids':        False,
        'plot_agg_line_installedCap_per_month':  False,
        'plot_agg_line_productionHOY_per_node':  False,
        'plot_agg_line_gridPremiumHOY_per_node': False,
        'plot_agg_line_gridpremium_structure':   False,
        'plot_agg_line_production_per_month':    False,
        'plot_agg_line_cont_charact_new_inst':   False,
    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
# if not not dataagg_scenarios:/
for k_sett, scen_sett in dataagg_scenarios.items():
    dataagg_settings = scen_sett
    data_aggregation_MASTER.data_aggregation_MASTER(dataagg_settings) if run_dataagg else print('')


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    pvalloc_initialization_MASTER.pvalloc_initialization_MASTER(pvalloc_settings) if run_alloc_init else print('')
    pvalloc_MCalgorithm_MASTER.pvalloc_MC_algorithm_MASTER(pvalloc_settings) if run_alloc_MCalg else print('')


# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings) if run_visual else print('')



# END ==========================================================================
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




