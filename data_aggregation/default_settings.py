import copy

dataagg_default_settings = {
        'name_dir_export': 'preprep_BSBLSO_18to22',             # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
        'script_run_on_server':     False,                      # F: run on private computer, T: run on server
        'smaller_import':           False,                      # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
        'show_debug_prints':        True,                       # F: certain print statements are omitted, T: includes print statements that help with debugging
        'turnoff_comp_after_run':   False,                      # F: keep computer running after script is finished, T: turn off computer after script is finished
        'wd_path_laptop': 'C:/Models/OptimalPV_RH',             # path to the working directory on Raul's laptop
        'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',  # path to the working directory on the server

        'kt_numbers': [11,12,13],                               # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
        'bfs_numbers': [],                                      # list of municipalites to select for allocation (only used if kt_numbers == 0)
        'year_range': [2021, 2022],                             # range of years to import
        
        # switch on/off parts of aggregation
        'split_data_geometry_AND_slow_api': True, 
        'rerun_localimport_and_mappings':   True,               # F: use existi ng parquet files, T: recreate parquet files in data prep
        'reextend_fixed_data':              True,               # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again       
        
        # settings for gwr selection
        'gwr_selection_specs': {
            'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                              'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                              'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
            'dwelling_cols': ['EGID', 'WAZIM', 'WAREA', ],
            'DEMAND_proxy': 'GAREA',
            'GSTAT': ['1004',],                                 # GSTAT - 1004: only existing, fully constructed buildings
            'GKLAS': ['1110','1121'], # ,'1276'],                # GKLAS - 1110: only 1 living space per building; 1121: Double-, row houses with each appartment (living unit) having it's own roof; 1276: structure for animal keeping (most likely still one owner)
            'GBAUJ_minmax': [1950, 2022],                       # GBAUJ_minmax: range of years of construction
            'GWAERZH': ['7410', '7411',],                       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
            # 'GENH': ['7580', '7581', '7582'],                 # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                                                # GANZWHG - total number of apartments in building
                                                                # GAZZI - total number of rooms in building
            },
        'solkat_selection_specs': {
            'col_partition_union': 'SB_UUID',                   # column name used for the union of partitions
            'GWR_EGID_buffer_size': 0.75,                          # buffer size in meters for the GWR selection
            }   
        }


def extend_dataag_scen_with_defaults(scen_dict, defaults = dataagg_default_settings):
    new_scen_dict = {}

    for scen_name, scen_sett in scen_dict.items():
        default_dict = copy.deepcopy(defaults)
        default_dict['name_dir_export'] = scen_name

        for k_sett, v_sett in scen_sett.items():
            if not isinstance(v_sett, dict):
                default_dict[k_sett] = v_sett

            elif isinstance(v_sett, dict) and k_sett in default_dict.keys():
                for k_sett_sub, v_sett_sub in v_sett.items():
                    default_dict[k_sett][k_sett_sub] = v_sett_sub
            
        new_scen_dict[scen_name] = default_dict

    return new_scen_dict


def get_default_dataag_settings():
    return dataagg_default_settings