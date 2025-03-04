import sys
import os as os
import glob
import datetime as datetime
from typing_extensions import List

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, get_bfs_from_ktnr

import split_data_geometry
import api_pvtarif
import get_elecpri_data
import preprep_data
import sql_gwr
import extend_data



# ------------------------
# *** DATA AGGREGATION ***
# ------------------------


class DataAggScenario:
    # DEFAULT SETTINGS ---------------------------------------------------
    def __init__(self, 
                 name_dir_export    = 'preprep_BSBLSO_18to22',             # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
                 smaller_import     = False,                      # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
                 show_debug_prints  = True,                       # F: certain print statements are omitted, T: includes print statements that help with debugging

                 kt_numbers     = [],                               # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
                 bfs_numbers    = [2761,],                                      # list of municipalites to select for allocation (only used if kt_numbers == 0)
                 year_range     = [2021, 2022],                             # range of years to import

                 split_data_geometry_AND_slow_api   = False,
                 rerun_localimport_and_mappings     = True,               # F: use existi ng parquet files, T: recreate parquet files in data prep
                 reextend_fixed_data                = True,               # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again

                 GWR_building_cols  = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE',
                                       'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 
                                       'GANZWHG','GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 
                                       'GEBF', 'GAREA'],
                 GWR_dwelling_cols  = ['EGID', 'EWID', 'WAZIM', 'WAREA', ],
                 GWR_DEMAND_proxy   = 'GAREA',
                 GWR_GSTAT          = ['1004',],                                 # GSTAT - 1004: only existing, fully constructed buildings
                 GWR_GKLAS          = ['1110','1121'], # ,'1276'],                # GKLAS - 1110: only 1 living space per building; 1121: Double-, row houses with each appartment (living unit) having it's own roof; 1276: structure for animal keeping (most likely still one owner)
                 GWR_GBAUJ_minmax   = [1920, 2022],                       # GBAUJ_minmax: range of years of construction
                 GWR_GWAERZH        = ['7410', '7411',],                       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings

                 SOLKAT_col_partition_union                 = 'SB_UUID',                   # column name used for the union of partitions
                 SOLKAT_GWR_EGID_buffer_size                = 10,                          # buffer size in meters for the GWR selection
                 SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique = False,
                 SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG', ],
                 SOLKAT_test_loop_optim_buff_size_TF        = False,
                 SOLKAT_test_loop_optim_buff_arang          = [0, 10, 0.1],

                 DEMAND_input_data_source = "NETFLEX", #    Netflex OR SwissStore
                 ):
        
        self.name_dir_export: str = name_dir_export
        self.smaller_import: bool = smaller_import
        self.show_debug_prints: bool = show_debug_prints
        
        self.kt_numbers: List[int] = kt_numbers
        # not in init because data_path and log_name needed to be defined
        self.bfs_numbers = bfs_numbers# self.bfs_numbers: List[int] = get_bfs_from_ktnr(kt_numbers, data_path, log_name) if kt_numbers != [] else bfs_numbers
        self.year_range: List[int] = year_range

        self.split_data_geometry_AND_slow_api: bool = split_data_geometry_AND_slow_api
        self.rerun_localimport_and_mappings: bool = rerun_localimport_and_mappings
        self.reextend_fixed_data: bool = reextend_fixed_data

        self.GWR_building_cols: List[str] = GWR_building_cols
        self.GWR_dwelling_cols: List[str] = GWR_dwelling_cols
        self.GWR_DEMAND_proxy: str = GWR_DEMAND_proxy
        self.GWR_GSTAT: List[str] = GWR_GSTAT
        self.GWR_GKLAS: List[str] = GWR_GKLAS
        self.GWR_GBAUJ_minmax: List[int] = GWR_GBAUJ_minmax
        self.GWR_GWAERZH: List[str] = GWR_GWAERZH

        self.SOLKAT_col_partition_union: str = SOLKAT_col_partition_union
        self.SOLKAT_GWR_EGID_buffer_size: int = SOLKAT_GWR_EGID_buffer_size
        self.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique: bool = SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique
        self.SOLKAT_cols_adjust_for_missEGIDs_to_solkat: List[str] = SOLKAT_cols_adjust_for_missEGIDs_to_solkat
        self.SOLKAT_test_loop_optim_buff_size_TF: bool = SOLKAT_test_loop_optim_buff_size_TF
        self.SOLKAT_test_loop_optim_buff_arang: List[float] = SOLKAT_test_loop_optim_buff_arang

        self.DEMAND_input_data_source: str = DEMAND_input_data_source

    def run_data_agg(self):
        """
            Input:
                (preprep data directory defined in the pv allocation scenario settings)
                dict: pvalloc_settings_func
                    > settings for pv allocation scenarios, for initalization and Monte Carlo iterations

            Output (no function return but export to dir):
                > directory renamed after scenario name (pvalloc_scenario), containing all data files form the INITIALIZATION of the pv allocation run.

            Description: 
                > Depending on the settings, certain steps of the model initalization can be run. (Debug function to only run certain steps, based on interim 
                    file exports to save time).
                > First the prepared data (geo and time series) from the preprep_[scenario] directory is imported (based on sencario selection criteria) 
                    and a topology is created (dict with EGID as keys, containing all relevant information for each individual house).
                > Then the a the future construction capacity for each month is defined (based on the scenario settings and past construction volume (kWP 
                    in the smaple area and time window)).
                > Next, the topology dictionary is transformed to a dataframe, to then be merged with the radiation time series. This step is necessary, as 
                    I consider individual roof parts for each hour of the year). The total radiation potential per roof partition is calculated. This huge data
                    frame is then partitioned into smaller subfiles to be "operatable" by my python IDE  economic components. This iterative subfile strucutre can
                    be "switched off" (set n houses per subfile large enough) for larger computers or high performance computing clusters.
                > The next scetion of the MASTER file runs a number of sanity checks on the initalization of the pv allocation run. 
                    - The first check runs the allocation algorithm (identical to later Monte Carlo iterations), to extract plots and visualiations, accessible already
                    after only a few monthly iterations. 
                    - Another check exports all the relevant data from the topo dict and the economic components for each house to an xlsx file for comparison. 
                    - Another check runs a simple check for multiple installations per EGID (which should not happen in the current model).
                > The final step is to copy all relevant files to the output directory, which is then renamed after the scenario name.

            """

        # SETUP ---------------------------------------------------
        self.wd_path = os.getcwd()
        self.data_path = os.path.join(self.wd_path, 'data')
        self.preprep_path = os.path.join(self.data_path, 'preprep', 'preprep_scen__temp_to_be_renamed')
        self.dir_move_to = os.path.join(self.data_path, 'preprep', self.name_dir_export)

        self.log_name = os.path.join(self.preprep_path, 'preprep_log.txt')
        self.summary_name = os.path.join(self.preprep_path, 'summary_data_selection_log.txt')

        self.bfs_numbers: List[int] = get_bfs_from_ktnr(self.kt_numbers, self.data_path, self.log_name) if self.kt_numbers != [] else self.bfs_numbers
        self.total_runtime_start = datetime.datetime.now()

        # create dir for export
        os.makedirs(self.preprep_path, exist_ok=True)

        # create log file
        chapter_to_logfile(f'start MAIN_data_aggregation for:{self.name_dir_export}', self.log_name, overwrite_file=True)
        subchapter_to_logfile('dataagg_settings', self.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.log_name)

        # create summary file
        chapter_to_logfile('OptimalPV - Sample Summary of Building Topology', self.summary_name, overwrite_file=True)
        chapter_to_logfile('data_aggregation', self.summary_name)


        # RUN DATA AGGREGATION ---------------------------------------------------
        if self.split_data_geometry_AND_slow_api:
            # split geom from data and import slow APIs
            subchapter_to_logfile('pre-prep data: SPLIT DATA GEOMETRY + IMPORT SLOW APIs', self.log_name)
            split_data_geometry.split_data_geometry(self)

            subchapter_to_logfile('pre-prep data: API GM by EWR MAPPING', self.log_name)
            api_pvtarif.api_pvtarif_gm_ewr_Mapping(self)
            
            subchapter_to_logfile('pre-prep data: API PVTARIF', self.log_name)
            api_pvtarif.api_pvtarif_data(self)

        
        subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', self.log_name)
        get_elecpri_data.get_elecpri_data_earlier_api_import(self)

        subchapter_to_logfile('pre-prep data: API INPUT DATA', self.log_name)
        preprep_data.get_earlier_api_import_data(self)


        if self.rerun_localimport_and_mappings:
            subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', self.log_name)
            sql_gwr.sql_gwr_data(self)

            subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', self.log_name)
            preprep_data.local_data_AND_spatial_mappings(self)

            subchapter_to_logfile('pre-prep data: IMPORT DEMAND TS + match series HOUSES', self.log_name)
            preprep_data.import_demand_TS_AND_match_households(self)

            subchapter_to_logfile('pre-prep data: IMPORT METEO SUNSHINE TS', self.log_name)
            preprep_data.import_meteo_data(self)

        if self.reextend_fixed_data:
            subchapter_to_logfile('extend data: GET ANGLE+TILT FACTOR + NODE MAPPING', self.log_name)
            extend_data.get_angle_tilt_table(self)

        
        # END + FOLDER RENAME ---------------------------------------------------
        chapter_to_logfile(f'END MASTER_data_aggregation\n Runtime (hh:mm:ss):{datetime.datetime.now() - self.total_runtime_start}', self.log_name)

        if os.path.exists(self.dir_move_to):
            n_same_names = len(glob.glob(f'{self.name_dir_export}*'))
            os.rename(self.dir_move_to, f'{self.dir_move_to}_{n_same_names}')

        os.rename(self.log_name, f'{self.log_name.split(".txt")[0]}_{self.name_dir_export}.txt')
        os.rename(self.summary_name, f'{self.summary_name.split(".txt")[0]}_{self.name_dir_export}.txt')
        
        # rename preprep folder
        os.rename(self.preprep_path, self.dir_move_to)

        

# ==================================================================================================================
if __name__ == '__main__':
    preprep_scen_list = [
        # DataAggScenario(
        #     name_dir_export = 'preprep_BLBSSO_18to23_1and2homes_API_reimport',
        #     kt_numbers = [11, 12, 13],
        #     year_range = [2018, 2023],
        #     split_data_geometry_AND_slow_api = True,
        #     GWR_GKLAS = ['1110', '1121', '1276'],
            
        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),
        DataAggScenario(
            name_dir_export = 'preprep_BL_22to23_extSolkatEGID_DFUIDduplicates',
            # kt_numbers = [13],
            bfs_numbers = [2761, 2768,],
            year_range = [2022, 2023],

            GWR_GKLAS = ['1110', '1121'],

            SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        ),

        # DataAggScenario(
        #     name_dir_export = 'preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates',
        #     kt_numbers = [11],
        #     year_range = [2022, 2023],

        #     GWR_GKLAS = ['1110', '1121'],

        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),

        # DataAggScenario(
        #     name_dir_export = 'preprep_BLBSSO_22to23_extSolkatEGID_DFUIDduplicates',
        #     kt_numbers = [13, 12, 11],
        #     year_range = [2022, 2023],

        #     GWR_GKLAS = ['1110', '1121'],

        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),
    ]

    for preprep_scen in preprep_scen_list:
        preprep_scen.run_data_agg()


print('done')

