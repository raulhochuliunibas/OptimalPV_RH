import os

import data_aggregation_MASTER
import pv_allocation_MASTER
import visualization_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
os.chdir('C:/Models/OptimalPV_RH')


# data_aggregation 
datagg_scenarios = {
    'preprep_BSBLSO_18to22':{
        'show_debug_prints': True,
        }, 
}

datagg_scenarios = extend_dataag_scen_with_defaults(datagg_scenarios)


# pv_allocation 
pvalloc_scenarios={
    'pvalloc_smallBL_SLCTN_npv_weighted': {
            'algorithm_specs': {'inst_selection_method': 'prob_weighted_npv',},
    },
    'pvalloc_smallBL_SLCTN_random': {
            'algorithm_specs': {'inst_selection_method': 'random',},
    },
    'pvalloc_smallBL_10y_npv_weighted': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',
            'months_prediction': 12*10,
            'recreate_topology':            True, 
            'recalc_economics_topo_df':     True,
            'run_allocation_loop':          True,

            'algorithm_specs': {'inst_selection_method': 'prob_weighted_npv',},
    },
    'pvalloc_smallBL_10y_random': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',    
            'months_prediction': 12*10,
            'recreate_topology':            True,
            'recalc_economics_topo_df':     True,
            'run_allocation_loop':          True,

            'algorithm_specs': {'inst_selection_method': 'random',},
    },
}

parkplatz = {
        'pvalloc_BL_SLCTN_random': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',
            'kt_numbers': [13, ],
            'recreate_topology':            True,
            'recalc_economics_topo_df':     True,
            'run_allocation_loop':          True,

            'algorithm_specs': {'inst_selection_method': 'random',      
                                'topo_subdf_partitioner': 500,
                                },
        }, 
        'pvalloc_BL_SLCTN_npv': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',
            'kt_numbers': [13, ],
            'recreate_topology':            True,
            'recalc_economics_topo_df':     True,
            'run_allocation_loop':          True,

            'algorithm_specs': {'inst_selection_method': 'prob_weighted_npv',
                                'topo_subdf_partitioner': 500,
                                },
        }
}

pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)
print(pvalloc_scenarios)


# vsualiastion 
visual_settings= {
    }



# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in datagg_scenarios.items():
    dataagg_settings = scen_sett
    # data_aggregation_MASTER.data_aggregation_MASTER(dataagg_settings)


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    pv_allocation_MASTER.pv_allocation_MASTER(pvalloc_settings)
    

# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings)



# END ==========================================================================
print(f'{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




# old settings dicts  >  to be deleted in Oct 2024
dataagg_settings = {
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

pvalloc_settings = {
                'name_dir_export': 'pvalloc_BL_smallsample',              # name of the directory where all proccessed data is stored at the end of the code file 
                'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h', # name of the directory where preprepared data is stored and accessed by the code
                'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
                'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

                'kt_numbers': [], # [13, ][11,12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain indiviual municipalities
                'bfs_numbers': [2791, 2784, 2781, 2789, 2782, 2793, 2787, 2792, 2613, 2614, 2476, 2477],
                'T0_prediction': '2023-01-01 00:00:00', 
                'months_lookback': 12*1,
                'months_prediction': 5,

                'script_run_on_server':     False,                           # F: run on private computer, T: run on server
                'show_debug_prints':        True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
                'fast_debug_run':           False,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
                'n_egid_in_topo': 200, 
                'recreate_topology':            True, 
                'recalc_economics_topo_df':     True,
                'run_allocation_loop':          False,

                'create_map_of_topology':       False,

                'recalc_npv_all_combinations':  True,
                'test_faster_if_subdf_deleted': False,
                'test_faster_npv_update_w_subdf_npry': True, 

                'algorithm_specs': {
                    'inst_selection_method': 'prob_weighted_npv', # random, prob_weighted_npv, 
                    'montecarlo_iterations': 1,
                    'keep_files_each_iterations': ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 
                    'keep_files_only_one': ['elecpri.parquet', 'pvtarif.parquet', 'pv.parquet', 'meteo_ts'],
                    'rand_seed': 42,                            # random seed set to int or None
                    'while_inst_counter_max': 5000,
                    'capacity_tweak_fact': 1, 
                    'topo_subdf_partitioner': 800,
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
