import copy

pvalloc_default_settings = {
    'name_dir_export': 'pvalloc_BL_smallsample',              # name of the directory where all proccessed data is stored at the end of the code file 
    'name_dir_import': 'preprep_BSBLSO_21to22_1and2homes', # name of the directory where preprepared data is stored and accessed by the code
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
    'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

    # main settings for allocation
    'kt_numbers': [], #[12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain indiviual municipalities
    'bfs_numbers': [2791, 2787, 2792, 2784, 2793, 2782, 2781, 2789, 2786, 2768, 2772, 2785, 2761],
    'T0_prediction': '2023-01-01 00:00:00', 
    'months_lookback': 12*1,
    'months_prediction': 3,
    'script_run_on_server':     False,                           # F: run on private computer, T: run on server
    'show_debug_prints':        True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
    'fast_debug_run':           False,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
    'n_egid_in_topo': 200, 


    # switch on/off parts of aggregation 
    'recreate_topology':             True, 
    'recalc_economics_topo_df':      True,
    'sanitycheck_byEGID':            True,
    'create_gdf_export_of_topology': True,

    # PART I: settings for alloc_initialization --------------------
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
    'weather_specs': {
        'meteo_col_radiation_proxy': ['Basel Direct Shortwave Radiation','Basel Diffuse Shortwave Radiation',],
        'meteo_col_temperature_proxy': ['Basel Temperature [2 m elevation corrected]',], 
        'diffuse_to_direct_rad_factor': 0.2,
        'weather_year': 2022,
    },
    'constr_capacity_specs': {
        'ann_capacity_growth': 0.05,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
        'constr_capa_overshoot_fact': 1, 
        'summer_months': [4,5,6,7,8,9,],
        'winter_months': [10,11,12,1,2,3,],
        'share_to_summer': 0.6, 
        'share_to_winter': 0.4,
    },
    'tech_economic_specs': {
        'self_consumption_ifapplicable': 0,
        'interest_rate': 0.01,
        'pvtarif_year': 2022, 
        'pvtarif_col': ['energy1', 'eco1'],
        'pvprod_calc_method': 'method2',
        'panel_efficiency': 0.15,         # XY% Wirkungsgrad PV Modul
        'inverter_efficiency': 0.95,        # XY% Wirkungsgrad Wechselrichter
        'elecpri_year': 2022,
        'elecpri_category': 'H4', 
        'invst_maturity': 25,
        'kWpeak_per_m2': 0.2,                       # A 1m2 area can fit 0.2 kWp of PV Panels, 10kWp per 50m2; ASSUMPTION HECTOR: 300 Wpeak / 1.6 m2
        'share_roof_area_available': 0.7,           # x% of the roof area is effectively available for PV installation  ASSUMPTION HECTOR: 70%¨
        'max_distance_m_for_EGID_node_matching': 0, # max distance in meters for matching GWR EGIDs that have no node assignment to the next grid node
        },
    'panel_efficiency_specs': {
        'variable_panel_efficiency_TF': True,
        'summer_months': [6,7,8,9],
        'hotsummer_hours': [11, 12, 13, 14, 15, 16, 17,],
        'hot_hours_discount': 0.1,
    },
    'sanitycheck_summary_byEGID_specs': {
        'egid_list': [ # ['3031017','1367570', '3030600',], # '1367570', '245017418'      # known houses in the sample
                    '391292', '390601', '2347595', '401781'        # single roof houses in Aesch, Ettingen, 
                    '391263', '245057295', '401753',  # houses with built pv in Aesch, Ettingen,
            ],
        'n_EGIDs_of_alloc_algorithm': 10,
        'n_iterations_before_sanitycheck': 12,
    },
    # PART II: settings for MC algorithm --------------------
    'MC_loop_specs': {
        'montecarlo_iterations': 2,
        'fresh_initial_files': 
            ['topo_egid.json', 'months_prediction.parquet', 'gridprem_ts.parquet', 
              'constrcapa.parquet', 'dsonodes_df.parquet'],  #'gridnode_df.parquet',
        'keep_files_month_iter_TF': True,
        'keep_files_month_iter_max': 8,
        
        'keep_files_month_iter_list': ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 
        # 'keep_files_only_one': ['elecpri.parquet', 'pvtarif.parquet', 'pv.parquet', 'meteo_ts'],

    },
    'algorithm_specs': {
        'inst_selection_method': 'prob_weighted_npv',   # random, prob_weighted_npv, max_npv 
        'rand_seed': 42,                                # random seed set to int or None
        'while_inst_counter_max': 5000,
        'topo_subdf_partitioner': 400,
        'npv_update_grouby_cols_topo_aggdf': 
            ['EGID', 'df_uid', 'grid_node', 'bfs', 'gklas', 'demandtype',
            'inst_TF', 'info_source', 'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
            'FLAECHE', 'FLAECH_angletilt', 'AUSRICHTUNG', 'NEIGUNG','STROMERTRAG'], 
        'npv_update_agg_cols_topo_aggdf': {
            'pvprod_kW': 'sum', 'demand_kW': 'sum', 'selfconsum_kW': 'sum', 'netdemand_kW': 'sum', 
            'netfeedin_kW': 'sum', 'econ_inc_chf': 'sum', 'econ_spend_chf': 'sum'}, 

        'tweak_constr_capacity_fact': 1,
        'tweak_npv_calc': 1,
        'tweak_npv_excl_elec_demand': True,
        'tweak_gridnode_df_prod_demand_fact': 1,
        'constr_capa_overshoot_fact':1, # not in that dir but should be a single tweak factor dict. 
    },  
    'gridprem_adjustment_specs': {
        'tier_description': 'tier_level: (voltage_threshold, gridprem_Rp_kWh)',
        'power_factor': 1, 
        'perf_factor_1kVA_to_XkW': 0.8,
        'colnames': ['tier_level', 'used_node_capa_rate', 'gridprem_Rp_kWh'],
        'tiers': { 
            1: [0.7, 1], 
            2: [0.8, 3],
            4: [0.9, 7],
            5: [0.95, 15], 
            6: [10, 100],
            },},

    # PART III: post processing of MC algorithm --------------------
    # ...

}


def extend_pvalloc_scen_with_defaults(scen_dict, defaults=pvalloc_default_settings):
    scen_dicts = {}

    for scen_name, scen_sett in scen_dict.items():
        default_dict = copy.deepcopy(defaults)
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

