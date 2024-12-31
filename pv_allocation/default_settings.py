import copy

pvalloc_default_settings = {
    'name_dir_export': 'pvalloc_BL_smallsample',              # name of the directory where all proccessed data is stored at the end of the code file 
    'name_dir_import': 'preprep_BLSO_22to23_1and2homes', # name of the directory where preprepared data is stored and accessed by the code
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
    'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

    # main settings for allocation
    'kt_numbers': [], #[11,13],                           # list of cantons to be considered [11, Solothurn, 12 Basel-Stadt, 13 Basel-Landschaft], 0 used for NON canton-selection, selecting only certain indiviual municipalities
    'bfs_numbers': [
        2768, 2761, 2772, 2785,                             # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen
        2473, 2475, 2480,                                   # SOsml: Dornach, Hochwald, Seewen
        2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,     # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
        2618, 2621, 2883, 2622, 2616,                       # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
    ],
    'T0_prediction': '2023-01-01 00:00:00', 
    'months_lookback': 12*1,
    'months_prediction': 12,
    'script_run_on_server':     False,                           # F: run on private computer, T: run on server
    'export_csvs':              False, 
    # no longer relevant
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
        'solkat_max_area_per_EGID': None,
        'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                    'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                    'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
        'dwelling_cols': None, # ['EGID', 'WAZIM', 'WAREA', ],
        'DEMAND_proxy': 'GAREA',
        'GSTAT': ['1004',],                 # GSTAT - 1004: only existing, fully constructed buildings
        'GKLAS': ['1110','1121'], #,'1276',],                 # GKLAS - 1110: only 1 living space per building
        'GBAUJ_minmax': [1950, 2022],       # GBAUJ_minmax: range of years of construction
        # 'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
        # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                            # GANZWHG - total number of apartments in building
                                            # GAZZI - total number of rooms in building
    },
    'weather_specs': {
        'meteo_col_dir_radiation': 'Basel Direct Shortwave Radiation',
        'meteo_col_diff_radiation': 'Basel Diffuse Shortwave Radiation',
        'meteo_col_temperature': 'Basel Temperature [2 m elevation corrected]', 
        'weather_year': 2022,

        'radiation_to_pvprod_method': 'dfuid_ind',        #'flat', 'dfuid_ind'
        'rad_rel_loc_max_by': 'dfuid_specific',                   # 'all_HOY', 'dfuid_specific'
        'flat_direct_rad_factor': 1,
        'flat_diffuse_rad_factor': 1,
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
        'pvprod_calc_method': 'method3',
        'panel_efficiency': 0.15,         # XY% Wirkungsgrad PV Modul
        'inverter_efficiency': 0.8,        # XY% Wirkungsgrad Wechselrichter
        'elecpri_year': 2022,
        'elecpri_category': 'H4', 
        'invst_maturity': 25,
        'kWpeak_per_m2': 0.2,                       # A 1m2 area can fit 0.2 kWp of PV Panels, 10kWp per 50m2; ASSUMPTION HECTOR: 300 Wpeak / 1.6 m2
        'share_roof_area_available': 1,           # x% of the roof area is effectively available for PV installation  ASSUMPTION HECTOR: 70%¨
        'max_distance_m_for_EGID_node_matching': 0, # max distance in meters for matching GWR EGIDs that have no node assignment to the next grid node
        },
    'panel_efficiency_specs': {
        'variable_panel_efficiency_TF': True,
        'summer_months': [6,7,8,9],
        'hotsummer_hours': [11, 12, 13, 14, 15, 16, 17,],
        'hot_hours_discount': 0.1,
    },
    'sanitycheck_summary_byEGID_specs': {
        'egid_list': [                                             # ['3031017','1367570', '3030600',], # '1367570', '245017418'      # known houses in the sample in Laufen
                    '391292', '390601', '2347595', '401781'        # single roof houses in Aesch, Ettingen, 
                    '391263', '245057295', '401753',               # houses with built pv in Aesch, Ettingen,
                    
                    '245054165','245054166','245054175','245060521', # EGID selection of neighborhood within Aesch to analyse closer
                    '391253','391255','391257','391258','391262',
                    '391263','391289','391290','391291','391292',
                    '245057295', '245057294', '245011456', '391379', '391377'
            ],
        'n_EGIDs_of_alloc_algorithm': 20,
        'n_iterations_before_sanitycheck': 12,
    },
    # PART II: settings for MC algorithm --------------------
    'MC_loop_specs': {
        'montecarlo_iterations': 1,
        'fresh_initial_files': 
            ['topo_egid.json', 'months_prediction.parquet', 'gridprem_ts.parquet', 
              'constrcapa.parquet', 'dsonodes_df.parquet'],  #'gridnode_df.parquet',
        'keep_files_month_iter_TF': True,
        'keep_files_month_iter_max': 8,
        
        'keep_files_month_iter_list': ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 
        # 'keep_files_only_one': ['elecpri.parquet', 'pvtarif.parquet', 'pv.parquet', 'meteo_ts'],

    },
    'algorithm_specs': {
        'inst_selection_method': 'random',   # random, prob_weighted_npv, max_npv 
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


[ 2768, 2761, 2772, 2473, 2475, 2785, 2480,] 
# Ettingen, Aesch, Pfeffingen, Dornach, Hochwald, Duggingen, Seewen 

# BFS_NUMMER Table - Basel Landschaft

BFS_NUMMER_BL = [
    2761,   # Aesch (BL)
    2762,   # Allschwil
    2841,   # Anwil
    2881,   # Arboldswil
    2821,   # Arisdorf
    2763,   # Arlesheim
    2822,   # Augst
    2842,   # Bennwil
    2764,   # Biel-Benken
    2765,   # Binningen
    2766,   # Birsfelden
    2781,   # Blauen
    2842,   # Böckten
    2767,   # Bottmingen
    2883,   # Bretzwil
    2782,   # Brislach
    2823,   # Bubendorf
    2843,   # Buckten
    2783,   # Burg im Leimental
    2844,   # Buus
    2884,   # Diegten
    2845,   # Diepflingen
    2784,   # Dittingen
    2785,   # Duggingen
    2885,   # Eptingen
    2768,   # Ettingen
    2824,   # Frenkendorf
    2825,   # Füllinsdorf
    2846,   # Gelterkinden
    2826,   # Giebenach
    2786,   # Grellingen
    2847,   # Häfelfingen
    2848,   # Hemmiken
    2827,   # Hersberg
    2886,   # Hölstein
    2849,   # Itingen
    2850,   # Känerkinden
    2851,   # Kilchberg (BL)
    2887,   # Lampenberg
    2888,   # Langenbruck
    2852,   # Läufelfingen
    2787,   # Laufen
    2828,   # Lausen
    2889,   # Lauwil
    2890,   # Liedertswil
    2788,   # Liesberg
    2829,   # Liestal
    2830,   # Lupsingen
    2853,   # Maisprach
    2769,   # Münchenstein
    2770,   # Muttenz
    2789,   # Nenzlingen
    2891,   # Niederdorf
    2854,   # Nusshof
    2892,   # Oberdorf (BL)
    2771,   # Oberwil (BL)
    2855,   # Oltingen
    2856,   # Ormalingen
    2772,   # Pfeffingen
    2831,   # Pratteln
    2832,   # Ramlinsburg
    2893,   # Reigoldswil
    2773,   # Reinach (BL)
    2857,   # Rickenbach (BL)
    2790,   # Roggenburg
    2791,   # Röschenz
    2858,   # Rothenfluh
    2859,   # Rümlingen
    2860,   # Rünenberg
    2774,   # Schönenbuch
    2833,   # Seltisberg
    2861,   # Sissach
    2862,   # Tecknau
    2863,   # Tenniken
    2775,   # Therwil
    2864,   # Thürnen
    2894,   # Titterten
    2792,   # Wahlen
    2895,   # Waldenburg
    2865,   # Wenslingen
    2866,   # Wintersingen
    2867,   # Wittinsburg
    2868,   # Zeglingen
    2834,   # Ziefen
    2869,   # Zunzgen
    2793,   # Zwingen

]

# BFS_NUMMER Table - Solothurn
BFS_NUMMER_SO = [
    2421,   # Aedermannsdorf 	          3
    2511,   # Aeschi (SO) 	              7
    2541,   # Balm bei Günsberg 	      3
    2422,   # Balsthal 	                  5
    2611,   # Bärschwil 	              1
    2471,   # Bättwil 	                  6
    2612,   # Beinwil (SO) 	              1
    2542,   # Bellach 	                  
    2543,   # Bettlach 	                  7
    2513,   # Biberist 	                  9
    2445,   # Biezwil 	                  3
    2514,   # Bolken 	                  4
    2571,   # Boningen 	                  2
    2613,   # Breitenbach 	              2
    2465,   # Buchegg 	                  9
    2472,   # Büren (SO) 	              6
    2614,   # Büsserach 	              4
    2572,   # Däniken 	                  
    2516,   # Deitingen 	              9
    2517,   # Derendingen 	              8
    2473,   # Dornach 	                  5
    2535,   # Drei Höfe 	              4
    2573,   # Dulliken 	                  8
    2401,   # Egerkingen 	              5
    2574,   # Eppenberg-Wöschnau 	      1
    2503,   # Erlinsbach (SO) 	          1
    2615,   # Erschwil 	                  5
    2518,   # Etziken 	                  5
    2616,   # Fehren 	                  
    2544,   # Feldbrunnen-St. Niklaus 	  8
    2545,   # Flumenthal 	              6
    2575,   # Fulenbach 	              5
    2474,   # Gempen 	                  1
    2519,   # Gerlafingen 	              6
    2546,   # Grenchen 	                  	702,6
    2576,   # Gretzenbach 	              3
    2617,   # Grindel 	                  5
    2547,   # Günsberg 	                  6
    2578,   # Gunzgen 	                  
    2579,   # Hägendorf 	              6
    2520,   # Halten 	                  3
    2402,   # Härkingen 	              7
    2491,   # Hauenstein-Ifenthal 	      2
    2424,   # Herbetswil 	              1
    2618,   # Himmelried 	              5
    2475,   # Hochwald 	                  7
    2476,   # Hofstetten-Flüh 	          3
    2425,   # Holderbank (SO) 	          8
    2523,   # Horriwil 	                  7
    2523,   # Horriwil 	                  7
    2548,   # Hubersdorf 	              4
    2524,   # Hüniken 	                  4
    2549,   # Kammersrohr 	              
    2580,   # Kappel (SO) 	              4
    2403,   # Kestenholz 	              5
    2492,   # Kienberg 	                  7
    2619,   # Kleinlützel 	              3
    2525,   # Kriegstetten 	              1
    2550,   # Langendorf 	              4
    2426,   # Laupersdorf 	              
    2526,   # Lohn-Ammannsegg             4
    2551,   # Lommiswil 	              5
    2493,   # Lostorf 	                  1
    2464,   # Lüsslingen-Nennigkofen 	  
    2527,   # Luterbach 	              
    2455,   # Lüterkofen-Ichertswil 	  9
    2427,   # Matzendorf 	              7
    2620,   # Meltingen 	              
    2457,   # Messen 	                  7
    2477,   # Metzerlen-Mariastein 	      5
    2428,   # Mümliswil-Ramiswil 	      2
    2404,   # Neuendorf 	              7
    2405,   # Niederbuchsiten 	          8
    2495,   # Niedergösgen 	              5
    2478,   # Nuglar-St. Pantaleon 	      8
    2621,   # Nunningen 	              5
    2406,   # Oberbuchsiten 	          9
    2553,   # Oberdorf (SO) 	          3
    2528,   # Obergerlafingen 	          3
    2497,   # Obergösgen 	              7
    2529,   # Oekingen 	                  2
    2407,   # Oensingen 	              8
    2581,   # Olten 	                  	1633,1
    2530,   # Recherswil 	              7
    2582,   # Rickenbach (SO) 	          1
    2554,   # Riedholz 	                  6
    2479,   # Rodersdorf 	              5
    2555,   # Rüttenen 	                  3
    2461,   # Schnottwil 	              6
    2583,   # Schönenwerd 	              6
    2480,   # Seewen 	                  9
    2556,   # Selzach 	                  2
    2601,   # Solothurn 	              	2696,8
    2584,   # Starrkirch-Wil 	          
    2499,   # Stüsslingen 	              8
    2532,   # Subingen 	                  4
    2500,   # Trimbach 	                  6
    2463,   # Unterramsern 	              9
    2585,   # Walterswil (SO) 	          8
    2586,   # Wangen bei Olten 	          7
    2430,   # Welschenrohr-Gänsbrunnen 	  3
    2501,   # Winznau 	                  4
    2502,   # Wisen (SO) 	              1
    2481,   # Witterswil 	              7
    2408,   # Wolfwil 	                  
    2534,   # Zuchwil 	                  3
    2622,   # Zullwil 	                  9 
]

