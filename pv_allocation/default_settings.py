pvalloc_default_settings = {
    'name_dir_export': 'pvalloc_BL_smallsample',              # name of the directory where all proccessed data is stored at the end of the code file 
    'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes', # name of the directory where preprepared data is stored and accessed by the code
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
    'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

    'kt_numbers': [], #[12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain indiviual municipalities
    'bfs_numbers': [2791, 2784, 2781, 2789, 2782, 2793, 2787, ], 
                    # 2792, 2613, 2614, 2476, 2477, 2481, 2471, 2768, 2761, 2772, 2786, 2785],
    'T0_prediction': '2023-01-01 00:00:00', 
    'months_lookback': 12*1,
    'months_prediction': 3,

    'script_run_on_server':     False,                           # F: run on private computer, T: run on server
    'show_debug_prints':        True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
    'fast_debug_run':           False,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
    'n_egid_in_topo': 200, 
    'recreate_topology':                True, 
    'recalc_economics_topo_df':         True,
    'create_gdf_export_of_topology':    False,
    'run_allocation_loop':              True,

    'recalc_npv_all_combinations':  True, 
    'test_faster_if_subdf_deleted': False,
    'test_faster_npv_update_w_subdf_npry': True, 

    'algorithm_specs': {
        'inst_selection_method': 'prob_weighted_npv', # random, prob_weighted_npv, max_npv 
        'montecarlo_iterations': 1,
        'keep_files_each_iterations': ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 
        'keep_files_only_one': ['elecpri.parquet', 'pvtarif.parquet', 'pv.parquet', 'meteo_ts'],
        'rand_seed': None,                            # random seed set to int or None
        'while_inst_counter_max': 5000,
        'topo_subdf_partitioner': 800,

        'tweak_capacity_fact': 1,
        'tweak_constr_capacity_fact': 1,
    },
    'gridprem_adjustment_specs': {
        'voltage_assumption': '',
        'tier_description': 'tier_level: (voltage_threshold, gridprem_plusRp_kWh)',
        'colnames': ['tier_level', 'vltg_threshold', 'gridprem_plusRp_kWh'],
        'tiers': { 
            1: [20, 1], 
            2: [40, 3],
            4: [60, 7],
            5: [80, 15], 
            6: [150, 50],
            },},
        # 'tiers': { 
        #     1: [2000, 1], 
        #     2: [4000, 3],
        #     4: [6000, 7],
        #     5: [8000, 15], 
        #     6: [15000, 50],
        #     },},
    'tech_economic_specs': {
        'self_consumption_ifapplicable': 1,
        'interest_rate': 0.01,
        'pvtarif_year': 2022, 
        'pvtarif_col': ['energy1', 'eco1'],
        'elecpri_year': 2022,
        'elecpri_category': 'H8', 
        'invst_maturity': 25,
        'conversion_m2tokW': 0.1,  # A 1m2 area can fit 0.1 kWp of PV Panels
    },
    'weather_specs': {
        'meteoblue_col_radiation_proxy': 'Basel Direct Shortwave Radiation',
        'weather_year': 2022,
    },
    'constr_capacity_specs': {
        'ann_capacity_growth': 0.1,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
        'summer_months': [4,5,6,7,8,9,],
        'winter_months': [10,11,12,1,2,3,],
        'share_to_summer': 0.6, 
        'share_to_winter': 0.4,
    },
    'gwr_selection_specs': {
        'solkat_max_n_partitions': 10,
        'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                    'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                    'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
        'dwelling_cols': None, # ['EGID', 'WAZIM', 'WAREA', ],
        'DEMAND_proxy': 'GAREA',
        'GSTAT': ['1004',],                 # GSTAT - 1004: only existing, fully constructed buildings
        'GKLAS': ['1110','1121','1276',],                 # GKLAS - 1110: only 1 living space per building
        'GBAUJ_minmax': [1950, 2022],       # GBAUJ_minmax: range of years of construction
        # 'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
        # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                            # GANZWHG - total number of apartments in building
                                            # GAZZI - total number of rooms in building
    },
}


def extend_pvalloc_scen_with_defaults(scen_dict, defaults=pvalloc_default_settings):
    scen_dicts = {}

    for scen_name, scen_sett in scen_dict.items():
        default_dict = defaults.copy()
        default_dict['name_dir_export'] = scen_name

        for k_sett, v_sett in scen_sett.items():
            if not isinstance(v_sett, dict):
                default_dict[k_sett] = v_sett

            elif isinstance(v_sett, dict) and k_sett in default_dict.keys():
                for k_sett_sub, v_sett_sub in v_sett.items():
                    default_dict[k_sett][k_sett_sub] = v_sett_sub
            
            scen_dicts[scen_name] = default_dict     

    return scen_dicts


def get_default_pvalloc_settings():
    return pvalloc_default_settings

