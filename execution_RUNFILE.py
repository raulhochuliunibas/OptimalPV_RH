import os



# SETTINGS DEFINITION =========================================================

# data_aggregation ------------------------------------------------------------
dataagg_settings = {
    {
        'name_dir_export': 'preprep_BSBLSO_18to22',     # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
        'script_run_on_server': False,                  # F: run on private computer, T: run on server
        'smaller_import': False,                	        # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
        'show_debug_prints': True,                      # F: certain print statements are omitted, T: includes print statements that help with debugging
        'turnoff_comp_after_run': False,                # F: keep computer running after script is finished, T: turn off computer after script is finished
        'wd_path_laptop': 'C:/Models/OptimalPV_RH',     # path to the working directory on Raul's laptop
        'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH', # path to the working directory on the server

        'kt_numbers': [11,12,13],                       # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
        'bfs_numbers': [],                              # list of municipalites to select for allocation (only used if kt_numbers == 0)
        'year_range': [2018, 2022],                     # range of years to import
        
        # switch on/off parts of aggregation
        'split_data_geometry_AND_slow_api': True, 
        # 'reimport_api_data_1': True,                   # F: use existing parquet files, T: recreate parquet files in data prep        
        'reimport_api_data': True,
        'rerun_localimport_and_mappings': True,         # F: use existi ng parquet files, T: recreate parquet files in data prep
        'reextend_fixed_data': True,                    # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again       
        
        # settings for gwr selection
        'gwr_selection_specs': {
            'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                        'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                        'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
            'dwelling_cols':['EGID', 'WAZIM', 'WAREA', ],
            'DEMAND_proxy': 'GAREA',
            'GSTAT': ['1004',],                 # GSTAT - 1004: only existing, fully constructed buildings
            'GKLAS': ['1110','1121','1276'],    # GKLAS - 1110: only 1 living space per building; 1121: Double-, row houses with each appartment (living unit) having it's own roof; 1276: structure for animal keeping (most likely still one owner)
            'GBAUJ_minmax': [1950, 2022],       # GBAUJ_minmax: range of years of construction
            'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
            # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                                # GANZWHG - total number of apartments in building
                                                # GAZZI - total number of rooms in building
            },
        'solkat_selection_specs': {
            'col_partition_union': 'SB_UUID',     # column name used for the union of partitions
            'GWR_EGID_buffer_size': 2,            # buffer size in meters for the GWR selection
            }   
        }
}


# pv_allocation ----------------------------------------------------------------
pvalloc_settings = {
    {
        'name_dir_export': 'pvalloc_BSBLSO_wrkn_prgrss',              # name of the directory where all proccessed data is stored at the end of the code file 
        'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h', # name of the directory where preprepared data is stored and accessed by the code
        'script_run_on_server': False,                           # F: run on private computer, T: run on server
        'fast_debug_run': True,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
        'show_debug_prints': True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
        'n_egid_in_topo': 150, 
        'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
        'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

        'kt_numbers': [13,], #[11,12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain indiviual municipalities
        'bfs_numbers': [2549, 2574, 2612, 2541, 2445, 2424, 2463, 2524, 2502, 2492], # list of bfs numbers to be considered
        
        # 'topology_year_range':[2019, 2022],
        # 'prediction_year_range':[2023, 2025],
        'T0_prediction': '2023-01-01 00:00:00', 
        'months_lookback': 12*1,
        'months_prediction': 12*2,
        'recreate_topology':            True, 
        'recalc_economics_topo_df':     True,
        'create_map_of_topology':       False,
        'recalc_npv_all_combinations':  True,

        'test_faster_if_subdf_deleted': False,

        'algorithm_specs': {
            'rand_seed': 42, 
            'safety_counter_max': 5000,
            'capacity_tweak_fact': 1, 
            'topo_subdf_partitioner': 1000,
        },
        'gridprem_adjustment_specs': {
            'voltage_assumption': '',
            'tier_description': 'tier_level: (voltage_threshold, gridprem_plusRp_kWh)',
            'colnames': ['tier_level', 'vltg_threshold', 'gridprem_plusRp_kWh'],
            'tiers': { 
                1: [200, 1], 
                2: [400, 3],
                4: [600, 7],
                5: [800, 15], 
                6: [1500, 50],
                },},
            # 'tiers': { 
            #     1: [2000, 1], 
            #     2: [4000, 3],
            #     4: [6000, 7],
            #     5: [8000, 15], 
            #     6: [15000, 50],
            #     },},
        'tech_economic_specs': {
            'interest_rate': 0.01,
            'pvtarif_year': 2022, 
            'pvtarif_col': ['energy1', 'eco1'],
            'elecpri_year': 2022,
            'elecpri_category': 'H8', 
            'invst_maturity': 25,
            'self_consumption_ifapplicable': 1,
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
        'assumed_parameters': {
        },

        'topo_type': 1,              # 1: all data, all egid  2: all data, only egid in solkat,  3: only partitions + Mappings, all egid, 4: only partitions + Mappings, only egid in solkat
        'rate_operation_cost': 0.01,                # assumed rate of operation cost (of investment cost)
        'NPV_include_wealth_tax': False,            # F: exclude wealth tax from NPV calculation, T: include wealth tax in NPV calculation
        'solkat_house_type_class': [0,],            # list of house type classes to be considered
        }
}



# EXECUTION ====================================================================


for setting in dataagg_settings:
    with open(f'{setting["wd_path_laptop"]}/data_aggregation_MASTER.py', 'r') as f:
        script_code = f.read()
    # exec(script_code)

for setting in pvalloc_settings:
    with open(f'{setting["wd_path_laptop"]}/pv_allocation_MASTER.py', 'r') as f:
        script_code = f.read()
    exec(script_code)