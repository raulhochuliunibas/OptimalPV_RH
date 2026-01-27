import sys
import os as os
import numpy as np
import glob
import datetime as datetime
import pandas as pd
import polars as pl
import geopandas as gpd
import  sqlite3
import shutil
import copy
import json
import requests
from shapely.geometry import Point
from shapely.ops import unary_union
import plotly.express as px
import itertools


from dataclasses import dataclass, field, asdict
from typing_extensions import List, Dict

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr
from src.api_keys import get_pvtarif_key, get_primeo_path



@dataclass
class DataAggScenario_Settings:
    # DEFAULT SETTINGS ---------------------------------------------------
    name_dir_export: str = 'preprep_BSBLSO_18to22'             # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
    smaller_import: bool = False                      # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
    show_debug_prints: bool = True                       # F: certain print statements are omitted, T: includes print statements that help with debugging

    kt_numbers: List[int] = field(default_factory=list)                               # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
    bfs_numbers: List[int] = field(default_factory=lambda: [2761,])                                      # list of municipalites to select for allocation (only used if kt_numbers == 0)
    year_range: List[int] = field(default_factory=lambda: [2021, 2022])                             # range of years to import

    split_data_geometry_AND_slow_api: bool = False
    rerun_localimport_and_mappings: bool = True               # F: use existi ng parquet files, T: recreate parquet files in data prep
    reextend_fixed_data: bool = True               # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again

    GWR_building_cols: List[str]    = field(default_factory=lambda: ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE',
                                            'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ',
                                            'GANZWHG',
                                            'GWAERZH1', 'GENH1',# 'GWAERSCEH1', 'GWAERDATH1',
                                            'GWAERZH2', 'GENH2',# 'GWAERSCEH2', 'GWAERDATH2',
                                            'GEBF', 'GAREA'])
    GWR_dwelling_cols: List[str]    = field(default_factory=lambda: ['EGID', 'EWID', 'WAZIM', 'WAREA', ])
    GWR_DEMAND_proxy: str           = 'GAREA'
    GWR_GSTAT: List[str]            = field(default_factory=lambda: [
                                            '1001', # GSTAT - 1001: in planing
                                            '1002', # GSTAT - 1002: construction right granted 
                                            '1003', # GSTAT - 1003: in construction
                                            '1004', # GSTAT - 1004: fully constructed, existing buildings
                                            ])                                 
    GWR_GKLAS: List[str]            = field(default_factory=lambda: [
                                            '1110', # GKLAS - 1110: only 1 living space per building
                                            '1121', # GKLAS - 1121: Double-, row houses with each appartment (living unit) having it's own roof;
                                            '1122', # GKLAS - 1122: Buildings with three or more appartments
                                            '1276', # GKLAS - 1276: structure for animal keeping (most likely still one owner)
                                            '1278', # GKLAS - 1278: structure for agricultural use (not anmial or plant keeping use, e.g. barns, machinery storage, silos),
                                            ])
    GWR_GBAUJ_minmax: List[int]     = field(default_factory=lambda: [1920, 2022])                       # GBAUJ_minmax: range of years of construction
    GWR_GWAERZH: List[str]          = field(default_factory=lambda: ['7410', '7411',])                       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
    GWR_AREtypology : Dict          = field(default_factory=lambda:  {
                                            'Urban': [1, 2, 4, ],
                                            'Suburban': [3, 5, 6 ], 
                                            'Rural': [7, 8,],                        
                                            # 1 - big centers   # https://map.geo.admin.ch/#/map?lang=en&center=2611872.51,1270543.42&z=3.703&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.vbs.schiessanzeigen,f;ch.astra.wanderland-sperrungen_umleitungen,f;ch.are.gemeindetypen;ch.swisstopo.swissboundaries3d-kanton-flaeche.fill&bgLayer=ch.swisstopo.pixelkarte-farbe            # '1' - big centers => URBAN
                                            # 2 - secondary centers of big centers  => URBAN 
                                            # 3 - crown big centers => SEMI-URBAN
                                            # 4 - medium centers => 
                                            # 5 - crown medium centers =>
                                            # 6 - small centers => 
                                            # 7 - peri-urban rural communes => RURAL
                                            # 8 - agricultural communes => RURAL
                                            # 9 - tourist communes => RURAL
    })
    GWR_SFHMFHtypology: Dict       = field(default_factory=lambda: {
                                            'SFH': ['1110', ],  # GKLAS - 1110: only 1 living space per building
                                            'MFH': [
                                                '1121',  # GKLAS - 1121: Double-, row houses with each appartment (living unit) having it's own roof;
                                                '1122',  # GKLAS - 1122: Buildings with three or more appartments
                                                '1276',  # GKLAS - 1276: structure for animal keeping (most likely still one owner)
                                                '1278',  # GKLAS - 1278: structure for agricultural use (not anmial or plant keeping use, e.g. barns, machinery storage, silos),
                                                ],
    })
    GWR_SFHMFH_outsample_proxy:str = 'MFH'



    SOLKAT_col_partition_union: str = 'SB_UUID'                   # column name used for the union of partitions
    SOLKAT_GWR_EGID_buffer_size: int = 10                          # buffer size in meters for the GWR selection
    SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique: bool = False
    SOLKAT_cols_adjust_for_missEGIDs_to_solkat: List[str] = field(default_factory=lambda: ['FLAECHE', 'STROMERTRAG', 'GSTRAHLUNG', 'MSTRAHLUNG' ])
    SOLKAT_test_loop_optim_buff_size_TF: bool = False
    SOLKAT_test_loop_optim_buff_arang: List[float] = field(default_factory=lambda: [0, 10, 0.1])

    DEMAND_input_data_source: str = 'SwissStore'#    # "NETFLEX"  OR "SwissStore"




class DataAggScenario:
    def __init__(self, settings: DataAggScenario_Settings):
        self.sett = settings
    
        self.sett.wd_path = os.getcwd()
        self.sett.data_path = os.path.join(self.sett.wd_path, 'data')
        self.sett.preprep_path = os.path.join(self.sett.data_path, 'preprep', 'preprep_scen__temp_to_be_renamed')
        self.sett.dir_move_to = os.path.join(self.sett.data_path, 'preprep', self.sett.name_dir_export)

    def export_dataagg_scen_settings(self):
            """
            Input:
                PVAllocScenario including the PVAllocScenario_Settings dataclass containing all scenarios settings. 
            (Output:)
                > Exports a JSON dict containing elements of the data class PVAllocScenario_Settings for pvallocation initialization.
                (all settings for the scenario).
            """
            sett_dict = asdict(self.sett)

            with open(f'{self.sett.preprep_path}/dataagg_sett.json', 'w') as f:
                json.dump(sett_dict, f, indent=4)

    def run_data_agg(self):
        """
            Input:
                - DataAggScenario_Settings
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
        self.sett.log_name = os.path.join(self.sett.preprep_path, 'preprep_log.txt')
        self.sett.summary_name = os.path.join(self.sett.preprep_path, 'summary_data_selection_log.txt')

        self.sett.bfs_numbers: List[str] = get_bfs_from_ktnr(self.sett.kt_numbers, self.sett.data_path, self.sett.log_name) if self.sett.kt_numbers != [] else [str(bfs) for bfs in self.sett.bfs_numbers]
        self.sett.total_runtime_start = datetime.datetime.now()

        # create dir for export
        os.makedirs(self.sett.preprep_path, exist_ok=True)

        # export class instance settings to dir
        self.export_dataagg_scen_settings()

        # create log file
        chapter_to_logfile(f'start MAIN_data_aggregation for: {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        subchapter_to_logfile('dataagg_settings', self.sett.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.sett.log_name)

        # create summary file
        chapter_to_logfile('OptimalPV - Sample Summary of Building Topology', self.sett.summary_name, overwrite_file=True)
        chapter_to_logfile('data_aggregation', self.sett.summary_name)


        # RUN DATA AGGREGATION ---------------------------------------------------
        if self.sett.split_data_geometry_AND_slow_api:
            subchapter_to_logfile('pre-prep data: SPLIT DATA GEOMETRY + IMPORT SLOW APIs', self.sett.log_name)
            self.split_data_geometry()

            subchapter_to_logfile('pre-prep data: API GM by EWR MAPPING', self.sett.log_name)
            self.api_pvtarif_gm_ewr_Mapping()
            
            subchapter_to_logfile('pre-prep data: API PVTARIF', self.sett.log_name)
            self.api_pvtarif_data()

        
        subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', self.sett.log_name)
        self.get_elecpri_earlier_api_import()

        subchapter_to_logfile('pre-prep data: API INPUT DATA', self.sett.log_name)
        self.get_preprep_data_earlier_api_import()


        if self.sett.rerun_localimport_and_mappings:
            subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', self.sett.log_name)
            self.sql_gwr_data()
            self.sql_gwr_ALL_CH_summary()

            subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', self.sett.log_name)
            self.preprep_local_data_AND_spatial_mappings()

            subchapter_to_logfile('pre-prep data: IMPORT DEMAND TS + match series HOUSES', self.sett.log_name)
            self.preprep_data_import_ts_AND_match_households()

            subchapter_to_logfile('pre-prep data: IMPORT METEO SUNSHINE TS', self.sett.log_name)
            self.preprep_data_import_meteo_data()

        if self.sett.reextend_fixed_data:
            subchapter_to_logfile('extend data: GET ANGLE+TILT FACTOR + NODE MAPPING', self.sett.log_name)
            self.extend_data_get_angle_tilt_table()

        
        # END + FOLDER RENAME ---------------------------------------------------
        chapter_to_logfile(f'end MAIN_data_aggregation\n Runtime (hh:mm:ss):{datetime.datetime.now() - self.sett.total_runtime_start}', self.sett.log_name)

        if os.path.exists(self.sett.dir_move_to):
            n_same_names = len(glob.glob(f'{self.sett.dir_move_to}*'))
            os.rename(self.sett.dir_move_to, f'{self.sett.dir_move_to}_{n_same_names}')

        os.rename(self.sett.log_name, f'{self.sett.log_name.split(".txt")[0]}_{self.sett.name_dir_export}.txt')
        os.rename(self.sett.summary_name, f'{self.sett.summary_name.split(".txt")[0]}_{self.sett.name_dir_export}.txt')
        
        # rename preprep folder
        os.rename(self.sett.preprep_path, self.sett.dir_move_to)



    # ======================================================================================================
    # ALL METHODS CALLED ABOVE in the MAIN METHODS
    # ======================================================================================================
    
    # DATA AGGREGATION ---------------------------------------------------------------------------
    if True: 
        def split_data_geometry(self):
            """
            Input:
                - DataAggScenario_Settings
            Tasks:
                - Split data and geometry for all geo data frames for faster importing later on
                - If required subset the data for the BSBLSO case (only for the selected cantons) for even faster import later
            Output to input_split_data_geometry:
                - subset of building (gwr), pv installations (pv) and roof partition (solkate) data for municipalities in BSBLSO case
            Output to preprep_data:
                - building (gwr), pv installations (pv) and roof partition (solkate) data for all selected municipalities
            """
            
            # SETUP --------------------------------------
            print_to_logfile('run function: split_data_and_geometry.py', self.sett.log_name)
            os.makedirs(f'{self.sett.data_path}/input_split_data_geometry', exist_ok=True)


            # IMPORT DATA --------------------------------------   
            gm_shp_df = gpd.read_file(f'{self.sett.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

            # Function: Merge GM BFS numbers to spatial data sources
            def attach_bfs_to_spatial_data(gdf, gm_shp_df, keep_cols = ['BFS_NUMMER', 'geometry' ]):
                """
                Function to attach BFS numbers to spatial data sources
                """
                gdf.set_crs(gm_shp_df.crs, allow_override=True, inplace=True)
                gdf = gpd.sjoin(gdf, gm_shp_df, how="left", predicate="within")
                dele_cols = ['index_right'] + [col for col in gm_shp_df.columns if col not in keep_cols]
                gdf.drop(columns = dele_cols, inplace = True)
                if 'BFS_NUMMER' in gdf.columns:
                    # transform BFS_NUMMER to str, np.nan to ''
                    gdf['BFS_NUMMER'] = gdf['BFS_NUMMER'].apply(lambda x: '' if pd.isna(x) else str(int(x)))

                return gdf


            # PV -------------------
            elec_prod_gdf = gpd.read_file(f'{self.sett.data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', layer ='ElectricityProductionPlant')
            pv_all_gdf = copy.deepcopy(elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'])
            checkpoint_to_logfile(f'import pv, {pv_all_gdf.shape[0]} rows', self.sett.log_name, 1, self.sett.show_debug_prints)

            pv_all_gdf = attach_bfs_to_spatial_data(pv_all_gdf, gm_shp_df)
            pv_all_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)

            # split + export
            checkpoint_to_logfile(f'-- check unique identifier pv: {pv_all_gdf["xtf_id"].nunique()} xtf unique, {pv_all_gdf.shape[0]} rows', self.sett.log_name, 0, self.sett.show_debug_prints)
            pv_pq = copy.deepcopy(pv_all_gdf.loc[:,pv_all_gdf.columns !='geometry'])
            pv_geo = copy.deepcopy(pv_all_gdf.loc[:,['xtf_id', 'BFS_NUMMER', 'geometry']])

            pv_pq.to_parquet(f'{self.sett.data_path}/input_split_data_geometry/pv_pq.parquet')
            checkpoint_to_logfile('-- exported pv_pq.parquet', self.sett.log_name, 5, self.sett.show_debug_prints)

            with open(f'{self.sett.data_path}/input_split_data_geometry/pv_geo.geojson', 'w') as f:
                f.write(pv_geo.to_json())
            checkpoint_to_logfile('-- exported pv_geo.geojson', self.sett.log_name, 5, self.sett.show_debug_prints)


            # SOLKAT -------------------
            solkat_all_gdf = gpd.read_file(f'{self.sett.data_path}/input\solarenergie-eignung-daecher_2056.gpkg\SOLKAT_DACH.gpkg', layer ='SOLKAT_CH_DACH')
            checkpoint_to_logfile(f'import solkat, {solkat_all_gdf.shape[0]} rows', self.sett.log_name, 2, self.sett.show_debug_prints)

            solkat_all_gdf = attach_bfs_to_spatial_data(solkat_all_gdf, gm_shp_df)
            solkat_all_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)

            # split + export
            checkpoint_to_logfile(f'-- check unique identifier solkat: {solkat_all_gdf["DF_UID"].nunique()} DF_UID unique, {solkat_all_gdf.shape[0]} rows', self.sett.log_name, 5, self.sett.show_debug_prints)
            solkat_pq = copy.deepcopy(solkat_all_gdf.loc[:,solkat_all_gdf.columns !='geometry'])
            solkat_geo = copy.deepcopy(solkat_all_gdf.loc[:,['DF_UID', 'BFS_NUMMER', 'geometry']])

            solkat_pq.to_parquet(f'{self.sett.data_path}/input_split_data_geometry/solkat_pq.parquet')
            checkpoint_to_logfile('-- exported solkat_pq.parquet', self.sett.log_name, 5, self.sett.show_debug_prints)

            with open(f'{self.sett.data_path}/input_split_data_geometry/solkat_geo.geojson', 'w') as f:
                f.write(solkat_geo.to_json())
            checkpoint_to_logfile('-- exported solkat_geo.geojson', self.sett.log_name, 5, self.sett.show_debug_prints)


            # SOLKAT MONTH -------------------
            solkat_month_pq = gpd.read_file(f'{self.sett.data_path}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='SOLKAT_CH_DACH_MONAT')
            solkat_month_pq.to_parquet(f'{self.sett.data_path}/input_split_data_geometry/solkat_month_pq.parquet')
            


            # SUBSET for BSBLSO case ========================================================
            bsblso_bfs_numbers = get_bfs_from_ktnr([11, 12, 13,], self.sett.data_path, self.sett.log_name)

            # PV -------------------
            checkpoint_to_logfile('subset pv for bsblso case', self.sett.log_name, 5, self.sett.show_debug_prints)
            pv_bsblso_geo = copy.deepcopy(pv_geo.loc[pv_geo['BFS_NUMMER'].isin(bsblso_bfs_numbers)])
            if pv_bsblso_geo.shape[0] > 0:
                with open (f'{self.sett.data_path}/input_split_data_geometry/pv_bsblso_geo.geojson', 'w') as f:
                    f.write(pv_bsblso_geo.to_json())
                checkpoint_to_logfile('-- exported pv_bsblso_geo.geojson', self.sett.log_name, 5, self.sett.show_debug_prints)

            # SOLKAT -------------------
            checkpoint_to_logfile('subset solkat for bsblso case', self.sett.log_name, 5, self.sett.show_debug_prints)
            solkat_bsblso_geo = copy.deepcopy(solkat_geo.loc[solkat_geo['BFS_NUMMER'].isin(bsblso_bfs_numbers)])
            if solkat_bsblso_geo.shape[0] > 0:
                with open (f'{self.sett.data_path}/input_split_data_geometry/solkat_bsblso_geo.geojson', 'w') as f:
                    f.write(solkat_bsblso_geo.to_json())
                checkpoint_to_logfile('-- exported solkat_bsblso_geo.geojson', self.sett.log_name, 5, self.sett.show_debug_prints)

            # GWR -------------------
            # get all BUILDING data 
            # select cols
            query_columns = self.sett.GWR_building_cols
            query_columns_str = ', '.join(query_columns)
            query_bfs_numbers = ', '.join([str(i) for i in bsblso_bfs_numbers])

            conn = sqlite3.connect(f'{self.sett.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
            sqlrows = cur.fetchall()
            conn.close()
            checkpoint_to_logfile('sql query ALL BUILDING done', self.sett.log_name, 5, self.sett.show_debug_prints)

            gwr_bsblso_pq = pd.DataFrame(sqlrows, columns=query_columns)

            # transform to gdf
            def gwr_to_gdf(df):
                df = df.loc[(df['GKODE'] != '') & (df['GKODN'] != '')]
                df[['GKODE', 'GKODN']] = df[['GKODE', 'GKODN']].astype(float)
                df['geometry'] = df.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
                gdf = gpd.GeoDataFrame(df, geometry='geometry')
                gdf.crs = 'EPSG:2056'
                return gdf
            gwr_bsblso_gdf = gwr_to_gdf(gwr_bsblso_pq)

            # export
            gwr_bsblso_pq.to_parquet(f'{self.sett.data_path}/input_split_data_geometry/gwr_bsblso_pq.parquet')
            gwr_bsblso_gdf.to_file(f'{self.sett.data_path}/input_split_data_geometry/gwr_bsblso_gdf.geojson', driver='GeoJSON')


            # Copy Log File to input_split_data_geometry folder
            if os.path.exists(self.sett.log_name):
                shutil.copy(self.sett.log_name, f'{self.sett.data_path}/input_split_data_geometry/split_data_geometry_logfile.txt')


        def api_pvtarif_gm_ewr_Mapping(self):
            '''
            Input:
                - DataAggScenario_Settings
            Tasks:
                - API call to get the mapping of the EWR (Electricity Distribution System Operator) for the selected municipalities in the BSBLSO case
            Output to preprep dir: 
                - Map_gm_ewr.parquet: mapping of the EWR for the selected municipalities in the BSBLSO case
            ''' 

            # SETUP --------------------------------------
            print_to_logfile('run function: api_pvtarif_gm_ewr_Mapping.py', self.sett.log_name)
            os.makedirs(f'{self.sett.data_path}/input_api', exist_ok=True)

            # QUERY --------------------------------------
            # gm_shp_df = gpd.read_file(f'{self.sett.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGRENZE.shp')
            gm_shp_df = gpd.read_file(f'{self.sett.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
            

            bfs_list = gm_shp_df['BFS_NUMMER'].unique()
            bfs_counter = len(bfs_list) // 4

            url = 'https://opendata.vese.ch/pvtarif/api/getData/muni'
            response_bfs_list = []
            Map_df = []
            checkpoint_to_logfile('api call pvtarif gm to ewr started', self.sett.log_name, 2,self.sett.show_debug_prints)

            for i_bfs, bfs in enumerate(bfs_list):
                req_url = f'{url}?idofs={bfs}&licenseKey={get_pvtarif_key()}'
                response = requests.get(req_url)
                response_json = response.json()

                if response_json['valid'] == True:
                    evus_list = response_json['evus']
                    sub_bfs_list = []
                    sub_nrElcom_list = []
                    sub_name_list = []
                    sub_idofs_list = []

                    for i in evus_list:
                        sub_bfs_list = sub_bfs_list+ [bfs]
                        sub_nrElcom_list = sub_nrElcom_list + [i['nrElcom']]
                        sub_name_list = sub_name_list + [i['Name']] 
                        sub_idofs_list = sub_idofs_list + [i['idofs']]

                    sub_Map_df = pd.DataFrame({'bfs': sub_bfs_list, 'nrElcom': sub_nrElcom_list, 'Name': sub_name_list, 'idofs': sub_idofs_list})
                    Map_df.append(sub_Map_df)
                    if i_bfs % bfs_counter == 0:
                        checkpoint_to_logfile(f'bfs: {bfs}, {i_bfs+1} of {len(bfs_list)} in list', self.sett.log_name, 3, self.sett.show_debug_prints)

            Map_gm_ewr = pd.concat(Map_df, ignore_index=True)
            checkpoint_to_logfile('api call pvtarif gm completed', self.sett.log_name, 3, self.sett.show_debug_prints)
            print_to_logfile(f'\n', self.sett.log_name)

            # EXPORT --------------------------------------
            Map_gm_ewr.to_parquet(f'{self.sett.data_path}/input_api/Map_gm_ewr.parquet')
            checkpoint_to_logfile('exported Map_gm_ewr from API', self.sett.log_name, 3)

            with open(f'{self.sett.data_path}/input_api/time_stamp.txt', 'w') as f:
                f.write(f'API call was run on : {pd.Timestamp.now()}')


        def api_pvtarif_data(self):
            '''
            Input:
                - DataAggScenario_Settings
            Tasks:
                - API call to get the PV compensation tariff for the selected municipalities 
                - This function imports the ID of all Distribution System Grid Operators of the VESE API (nrElcom) and 
                  their PV compensation tariff.
                - The data is aggregated by DSO and year and saved as parquet file in the data folder.
            Output to preprep dir:
                - pvtarif.parquet: PV compensation tariff for the selected municipalities
            '''
            # SETUP --------------------------------------
            print_to_logfile('run function: api_pvtarif.py', self.sett.log_name)
            os.makedirs(f'{self.sett.data_path}/input_api', exist_ok=True)

            # QUERY --------------------------------------
            year_range_list = [str(year % 100).zfill(2) for year in range(self.sett.year_range[0], self.sett.year_range[1]+1)]
            response_all_df_list = []

            Map_gm_ewr = pd.read_parquet(f'{self.sett.data_path}/input_api/Map_gm_ewr.parquet')
            ew_id = Map_gm_ewr['nrElcom'].unique()

            ew_id_counter = len(ew_id) / 4

            url = "https://opendata.vese.ch/pvtarif/api/getData/evu?"

            for y in year_range_list:
                checkpoint_to_logfile(f'start api call pvtarif for year: {y}', self.sett.log_name, 3, self.sett.show_debug_prints)
                response_ew_list = []

                for i_ew, ew in enumerate(ew_id):
                    req_url = f'{url}evuId={ew}&year={y}&licenseKey={get_pvtarif_key()}'
                    response = requests.get(req_url)
                    response_json = response.json()

                    if response_json['valid'] == True:
                        response_ew_list.append(response_json)

                    # if i_ew % ew_id_counter == 0:
                    #     checkpoint_to_logfile(f'-- year: {y}, ew: {ew}, {i_ew+1} of {len(ew_id)} in list', log_file_name_def=log_file_name_def, n_tabs_def = 2, show_debug_prints_def=show_debug_prints_def)

                response_ew_df = pd.DataFrame(response_ew_list)
                response_ew_df['year'] = y
                response_all_df_list.append(response_ew_df)
                checkpoint_to_logfile(f'call year: {y} completed', self.sett.log_name, 3, self.sett.show_debug_prints)

            pvtarif_raw = pd.concat(response_all_df_list)
            checkpoint_to_logfile('api call pvtarif completed', self.sett.log_name, 3, self.sett.show_debug_prints)

            pvtarif = copy.deepcopy(pvtarif_raw)
            empty_cols = [col for col in pvtarif.columns if (pvtarif[col]=='').all()]
            pvtarif = pvtarif.drop(columns=empty_cols)

            # EXPORT --------------------------------------
            pvtarif.to_parquet(f'{self.sett.data_path}/input_api/pvtarif.parquet')
            checkpoint_to_logfile('exported pvtarif from API', self.sett.log_name, 3)


        def get_elecpri_earlier_api_import(self):
            """
            Input: 
                - DataAggScenario_Settings
            Tasks:
                - Get electricity prices from former api import
            Output to preprep dir:
                - elecpri.parquet: electricity prices for the selected municipalities
            """

            # SETUP --------------------------------------
            print_to_logfile(f'run function: get_elecpri_data_earlier_api_import', self.sett.log_name)

            # IMPORT + SUBSET DATA --------------------------------------
            # elecpri_all = pd.read_parquet(f'{self.sett.data_path}/input_api/elecpri.parquet')
            elecpri_all = pd.read_parquet(f'{self.sett.data_path}/input/ElCom_consum_price_api_data/elecpri.parquet')
            elecpri = elecpri_all.loc[elecpri_all['bfs_number'].isin(self.sett.bfs_numbers)]

            # EXPORT --------------------------------------
            checkpoint_to_logfile('export elecpri of local data from former api import', self.sett.log_name)
            elecpri.to_parquet(f'{self.sett.preprep_path}/elecpri.parquet')  


        def get_preprep_data_earlier_api_import(self):
            """
            Input:
                - DataAggScenario_Settings
            Tasks:
                - Function to import all api input data, previously downloaded and stored through various API calls
            Output to preprep dir:
                - Map_gm_ewr.parquet: mapping of the EWR for the selected municipalities in the BSBLSO case
                - pvtarif.parquet: PV compensation tariff for the selected municipalities
            """
            # SETUP --------------------------------------
            print_to_logfile('run function: get_earlier_api_import_data.py', self.sett.log_name)

            # IMPORT + STORE DATA in preprep folder --------------------------------------
            # Map_gm_ewr
            Map_gm_ewr = pd.read_parquet(f'{self.sett.data_path}/input_api/Map_gm_ewr.parquet')
            Map_gm_ewr.to_parquet(f'{self.sett.preprep_path}/Map_gm_ewr.parquet')
            Map_gm_ewr.to_csv(f'{self.sett.preprep_path}/Map_gm_ewr.csv', sep=';', index=False)
            checkpoint_to_logfile('Map_gm_ewr stored in prepreped data', self.sett.log_name, 2, self.sett.show_debug_prints)
            
            # pvtarif
            pvtarif_all = pd.read_parquet(f'{self.sett.data_path}/input_api/pvtarif.parquet')
            year_range_2int = [str(year % 100).zfill(2) for year in range(self.sett.year_range[0], self.sett.year_range[1]+1)]
            pvtarif = copy.deepcopy(pvtarif_all.loc[pvtarif_all['year'].isin(year_range_2int), :])
            pvtarif.to_parquet(f'{self.sett.preprep_path}/pvtarif.parquet')
            pvtarif.to_csv(f'{self.sett.preprep_path}/pvtarif.csv', sep=';', index=False)
            checkpoint_to_logfile('pvtarif stored in prepreped data', self.sett.log_name, 2, self.sett.show_debug_prints)
                

        def sql_gwr_data(self):
            """
            Input:
                - DataAggScenario_Settings
            Tasks:
                - Function to import data from the Building and Dwelling (Gebaeude und Wohungsregister) database.
                - Import data from SQL file and save the relevant variables locally as parquet file.
            Output to preprep dir:
                - gwr_mrg_all_buildling_in_bfs.parquet
                - gwr_all_buildings_df.parquet: all building data for the selected municipalities
                - gwr.parquet: merged data of all buildings given selection criteria

                - gwr_all_buildings_gdf.geojson: all building data for the selected municipalities in geojson format
                - gwr_gdf.geojson: merged data of all buildings given selection criteria in geojson format
            """ 
            # SETUP --------------------------------------
            print_to_logfile('run function: sql_gwr_data.py', self.sett.log_name)


            # QUERYs --------------------------------------

            # get DWELLING data
            # select cols
            query_columns = self.sett.GWR_dwelling_cols
            query_columns_str = ', '.join(query_columns)
            query_bfs_numbers = ', '.join([str(i) for i in self.sett.bfs_numbers])

            conn = sqlite3.connect(f'{self.sett.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM dwelling')
            sqlrows = cur.fetchall()
            conn.close()
            checkpoint_to_logfile('sql query DWELLING done', self.sett.log_name, 10, self.sett.show_debug_prints)

            gwr_dwelling_df = pd.DataFrame(sqlrows, columns=query_columns)
            gwr_dwelling_df[['WAZIM', 'WAREA']] = gwr_dwelling_df[['WAZIM', 'WAREA']].replace('', 0).astype(float)
            gwr_dwelling_df.to_csv(f'{self.sett.preprep_path}/gwr_dwelling_df.csv', sep=';', index=False)


            # get ALL BUILDING data
            # select cols
            query_columns = self.sett.GWR_building_cols
            query_columns_str = ', '.join(query_columns)
            query_bfs_numbers = ', '.join([str(i) for i in self.sett.bfs_numbers])

            conn = sqlite3.connect(f'{self.sett.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
            sqlrows = cur.fetchall()
            conn.close()
            checkpoint_to_logfile('sql query ALL BUILDING done', self.sett.log_name, 10, self.sett.show_debug_prints)

            gwr_all_building_df = pd.DataFrame(sqlrows, columns=query_columns)
            gwr_all_building_df.to_csv(f'{self.sett.preprep_path}/gwr_all_building_df.csv', sep=';', index=False)
            gwr_all_building_df.to_parquet(f'{self.sett.preprep_path}/gwr_all_building_df.parquet')


            # merger -------------------
            # gwr = gwr_building_df.merge(gwr_dwelling_df, on='EGID', how='left')
            gwr_mrg = gwr_all_building_df.merge(gwr_dwelling_df, on='EGID', how='left')


            # aggregate dwelling data per EGID -------------------
            print('print to log_file')
            print_to_logfile('aggregate dwelling data per EGID', self.sett.log_name)
            checkpoint_to_logfile(f'check gwr BEFORE aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', self.sett.log_name, 3, True)

            print('print to summary_file')
            print_to_logfile('aggregate dwelling data per EGID', self.sett.summary_name)
            checkpoint_to_logfile(f'check gwr BEFORE aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', self.sett.summary_name, 3, True)

            bldg_agg_cols = copy.deepcopy(self.sett.GWR_building_cols)
            bldg_agg_cols.remove('EGID')
            bldg_agg_meth = {col: 'first' for col in bldg_agg_cols}

            gwr_mrg['nEWID'] = gwr_mrg['EWID']
            def concat_strings(x):
                return '_'.join(x.dropna().astype(str))
            dwel_agg_meth = {'EWID':concat_strings,'nEWID': 'count', 'WAZIM': 'sum', 'WAREA': 'sum'}

            agg_meth = {**bldg_agg_meth, **dwel_agg_meth}
            gwr_mrg_after_agg =           gwr_mrg.groupby('EGID').agg(agg_meth).reset_index()
            gwr_mrg_all_building_in_bfs = gwr_mrg.groupby('EGID').agg(agg_meth).reset_index()
            gwr_mrg = copy.deepcopy(gwr_mrg_after_agg)

            checkpoint_to_logfile(f'check gwr AFTER aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', self.sett.log_name, 3, True)
            checkpoint_to_logfile(f'check gwr AFTER aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', self.sett.summary_name, 3, True)


            # filter for specs -------------------
            checkpoint_to_logfile(f'check gwr_mrg BEFORE filtering: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique() )/gwr_mrg.shape[0],2)*100} %', self.sett.log_name, 3, True)

            gwr_mrg0 = copy.deepcopy(gwr_mrg)
            gwr_mrg0['GBAUJ'] = gwr_mrg0['GBAUJ'].replace('', 0).astype(int)
            gwr_mrg1 = gwr_mrg0[(gwr_mrg0['GSTAT'].isin(self.sett.GWR_GSTAT))]
            gwr_mrg2 = gwr_mrg1[(gwr_mrg1['GKLAS'].isin(self.sett.GWR_GKLAS))]
            gwr_mrg3 = gwr_mrg2[(gwr_mrg2['GBAUJ'] >= self.sett.GWR_GBAUJ_minmax[0]) & (gwr_mrg2['GBAUJ'] <= self.sett.GWR_GBAUJ_minmax[1])]
            gwr = copy.deepcopy(gwr_mrg3)
            checkpoint_to_logfile(f'check gwr AFTER filtering: {gwr["EGID"].nunique()} unique EGIDs in gwr.shape {gwr.shape}, {round((gwr["EGID"].nunique() )/gwr_mrg.shape[0],2)*100} %', self.sett.log_name, 3, True)
            print_to_logfile('\n', self.sett.summary_name)


            # summary log -------------------
            print_to_logfile('Building and Dwelling data import:', self.sett.summary_name)
            checkpoint_to_logfile(f'gwr_mrg0.shape: {gwr_mrg0.shape}, EGID: {gwr_mrg0["EGID"].nunique()}', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'\t> selection range n BFS municipalities: {len(self.sett.bfs_numbers)}', self.sett.summary_name, 2, True)
            # print_to_logfile(f'\n', log_file_name_def=summary_name)
            checkpoint_to_logfile(f'after GSTAT selection, gwr.shape: {gwr_mrg1.shape} EGID.nunique: {gwr_mrg1["EGID"].nunique()} ({round((gwr_mrg1.shape[0] ) / gwr_mrg0.shape[0] * 100, 2)} % of gwr_mrg0)', self.sett.summary_name, 2, True) 
            checkpoint_to_logfile(f'\t> selection GSTAT: {self.sett.GWR_GSTAT} "only existing bulidings"', self.sett.summary_name, 2, True)
            # print_to_logfile(f'\n', log_file_name_def=summary_name)
            checkpoint_to_logfile(f'after GKLAS selection, gwr.shape: {gwr_mrg2.shape} EGID.nunique: {gwr_mrg2["EGID"].nunique()} ({round((gwr_mrg2.shape[0] ) / gwr_mrg1.shape[0] * 100, 2)} % of gwr_mrg1)', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'\t> selection GKLAS: {self.sett.GWR_GKLAS} "1110 - building w 1 living space, 1121 - w 2 living spaces, 1276 - agricluture buildings (stables, barns )"', self.sett.summary_name, 2, True)
            # print_to_logfile(f'\n', log_file_name_def=summary_name)
            checkpoint_to_logfile(f'after GBAUJ_minmax selection, gwr.shape: {gwr_mrg3.shape} EGID.nunique: {gwr_mrg3["EGID"].nunique()} ({round((gwr_mrg3.shape[0] ) / gwr_mrg2.shape[0] * 100, 2)} % of gwr_mrg3)', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'\t> selection GBAUJ_minmax: {self.sett.GWR_GBAUJ_minmax} "built construction between years"', self.sett.summary_name, 2, True)
            # print_to_logfile(f'\n', log_file_name_def=summary_name)
            checkpoint_to_logfile(f'from ALL gwr_mrg0 (aggregated with dwelling, bfs already selected): {gwr_mrg0["EGID"].nunique()-gwr_mrg3["EGID"].nunique()} of {gwr_mrg0["EGID"].nunique()} EGIDs removed ({round((gwr_mrg0["EGID"].nunique() - gwr_mrg3["EGID"].nunique()  )/gwr_mrg0["EGID"].nunique()*100, 2)}%, mrg0-mrg3 of mrg0)', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'\t> {gwr_mrg3["EGID"].nunique()} gwr_mrg3 EGIDS {round((gwr_mrg3["EGID"].nunique())/gwr_mrg0["EGID"].nunique()*100, 2)}%  of  {gwr_mrg0["EGID"].nunique()} gwr_mrg0', self.sett.summary_name, 2, True)
            print_to_logfile('\n', self.sett.summary_name)


            # check proxy possiblity -------------------
            # checkpoint_to_logfile(f'gwr_guilding_df.shape: {gwr_building_df.shape}, EGID: {gwr_building_df["EGID"].nunique()};\n  gwr_dwelling_df.shape: {gwr_dwelling_df.shape}, EGID: {gwr_dwelling_df["EGID"].nunique()};\n  gwr.shape: {gwr.shape}, EGID: {gwr["EGID"].nunique()}', self.sett.log_name, 2, True)
            
            checkpoint_to_logfile(f'* check for WAZIM: {gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAZIM", {round((gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', self.sett.log_name, 1, True)
            checkpoint_to_logfile(f'* check for WAREA: {gwr.loc[gwr["WAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAREA", {round((gwr.loc[gwr["WAREA"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', self.sett.log_name, 1, True)
            checkpoint_to_logfile(f'* check for GAREA: {gwr.loc[gwr["GAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty GAREA", {round((gwr.loc[gwr["GAREA"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', self.sett.log_name, 1, True)

            # checkpoint_to_logfile('Did NOT us a combination of building and dwelling data, \n because they overlap way too little. This makes sense \nintuitievly as single unit houses probably are not registered \nas dwellings in the data base.', log_file_name_def, 1, True)


            # merge dfs and export -------------------
            # gwr = gwr_building_df
            gwr.to_csv(f'{self.sett.preprep_path}/gwr.csv', sep=';', index=False)
            gwr_mrg_all_building_in_bfs.to_csv(f'{self.sett.preprep_path}/gwr_mrg_all_building_in_bfs.csv', sep=';', index=False)
            gwr.to_parquet(f'{self.sett.preprep_path}/gwr.parquet')
            gwr_mrg_all_building_in_bfs.to_parquet(f'{self.sett.preprep_path}/gwr_mrg_all_building_in_bfs.parquet')
            checkpoint_to_logfile('exported gwr data', self.sett.log_name, n_tabs_def = 4)


            # create spatial df and export -------------------
            def gwr_to_gdf(df):
                df = df.loc[(df['GKODE'] != '') & (df['GKODN'] != '')]
                df[['GKODE', 'GKODN']] = df[['GKODE', 'GKODN']].astype(float)
                df['geometry'] = df.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
                gdf = gpd.GeoDataFrame(df, geometry='geometry')
                gdf.crs = 'EPSG:2056'
                return gdf

            # gwr_gdf (will later be reimported and reexported again just because in preprep_data, all major geo spatial dfs are imported and exported)    
            gwr_gdf = gwr_to_gdf(gwr)
            gwr_gdf = gwr_gdf.loc[:, ['EGID', 'geometry']]
            gwr_gdf.to_file(f'{self.sett.preprep_path}/gwr_gdf.geojson', driver='GeoJSON')

            # gwr_all_building_gdf exported for DSO nodes location determination later
            gwr_all_building_gdf = gwr_to_gdf(gwr_all_building_df)
            gwr_all_building_gdf.to_file(f'{self.sett.preprep_path}/gwr_all_building_gdf.geojson', driver='GeoJSON')

            if self.sett.split_data_geometry_AND_slow_api:
                gwr_gdf.to_file(f'{self.sett.data_path}/input_split_data_geometry/gwr_gdf.geojson', driver='GeoJSON')


        def sql_gwr_ALL_CH_summary(self): 
            """
            Input:
                - DataAggScenario_Settings
            Tasks:
                - Function to import all buidlings from GWR (no BFS selection) and create summary statistics for CH wide coverage
            Output to preprep dir:
                - gwr_all_ch_summary.txt or .json or csv
            """
            # SETUP --------------------------------------
            print_to_logfile('run function: sql_gwr_ALL_CH_summary.py', self.sett.log_name)


            # QUERYs --------------------------------------

            # get ALL BUILDING data
            # select cols
            query_columns = self.sett.GWR_building_cols
            query_columns_str = ', '.join(query_columns)
            query_bfs_numbers = ', '.join([str(i) for i in self.sett.bfs_numbers])

            conn = sqlite3.connect(f'{self.sett.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM building')
            sqlrows = cur.fetchall()
            conn.close()

            gwr_allch_raw = pl.DataFrame(sqlrows, schema = query_columns)


            # AGGREGATION + EXPORT --------------------------------------
            gwr_allch_raw = gwr_allch_raw.rename({'GGDENR':'BFS_NUMMER'})
            gwr_allch_summary = gwr_allch_raw.group_by(['BFS_NUMMER', 'GSTAT', 'GKLAS']).agg([
                pl.col('EGID').count().alias('nEGID'),
                pl.col('GAREA').sum().alias('GAREA'),
            ])

            gwr_allch_summary.write_csv(f'{self.sett.preprep_path}/gwr_all_ch_summary.csv')
            gwr_allch_summary.write_parquet(f'{self.sett.preprep_path}/gwr_all_ch_summary.parquet')       



        def preprep_local_data_AND_spatial_mappings(self):
            """
            Input: 
                - DataAggScenario_Settings
            Tasks:
                - Function to import all the local data sources, remove and transform data where necessary and store only
                  the required data that is in range with the BFS municipality selection. 
                - When applicable, create mapping files, so that so that different data sets can be matched and spatial 
                  data can be reidentified to their geometry if necessary. 
            Output to preprep dir:
                - list = ['gm_shp_gdf', 'pv_gdf', 'solkat_gdf', 'gwr_gdf','gwr_buff_gdf', 'gwr_all_building_gdf', 
                          'omitt_gwregid_gdf', 'omitt_solkat_gdf', 'omitt_pv_gdf', 'omitt_gwregid_gridnode_gdf' ]
            """

            # SETUP --------------------------------------
            print_to_logfile('run function: local_data_AND_spatial_mappings.py', self.sett.log_name)

            # IMPORT DATA ---------------------------------------------------------------------------------
            gm_shp_gdf = gpd.read_file(f'{self.sett.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
            

            # PV ====================
            pv_all_pq = pd.read_parquet(f'{self.sett.data_path}/input_split_data_geometry/pv_pq.parquet')
            checkpoint_to_logfile(f'import pv_pq, {pv_all_pq.shape[0]} rows', self.sett.log_name, 2, self.sett.show_debug_prints)
            pv_all_geo = gpd.read_file(f'{self.sett.data_path}/input_split_data_geometry/pv_geo.geojson')
            checkpoint_to_logfile(f'import pv_geo, {pv_all_geo.shape[0]} rows', self.sett.log_name, 2, self.sett.show_debug_prints)

            # transformations
            pv_all_pq['xtf_id'] = pv_all_pq['xtf_id'].astype(str)
            pv_all_geo['xtf_id'] = pv_all_geo['xtf_id'].astype(str)

            pv = pv_all_pq[pv_all_pq['BFS_NUMMER'].isin(self.sett.bfs_numbers)]  # select and export pq for BFS numbers
            pv_wgeo = pv.merge(pv_all_geo[['xtf_id', 'geometry']], how = 'left', on = 'xtf_id') # merge geometry for later use
            pv_gdf = gpd.GeoDataFrame(pv_wgeo, geometry='geometry')


            # GWR ====================
            gwr = pd.read_parquet(f'{self.sett.preprep_path}/gwr.parquet')
            gwr_gdf = gpd.read_file(f'{self.sett.preprep_path}/gwr_gdf.geojson')
            gwr_all_building_gdf = gpd.read_file(f'{self.sett.preprep_path}/gwr_all_building_gdf.geojson')
            checkpoint_to_logfile(f'import gwr, {gwr.shape[0]} rows', self.sett.log_name, 5, self.sett.show_debug_prints)


            # SOLKAT ====================
            solkat_all_pq = pd.read_parquet(f'{self.sett.data_path}/input_split_data_geometry/solkat_pq.parquet')
            checkpoint_to_logfile(f'import solkat_pq, {solkat_all_pq.shape[0]} rows', self.sett.log_name,  1, self.sett.show_debug_prints)

            bsblso_bfs_numbers = get_bfs_from_ktnr([11,12,13], self.sett.data_path, self.sett.log_name)
            bsblso_bfs_numbers_TF = all([bfs in bsblso_bfs_numbers for bfs in self.sett.bfs_numbers])
            if (bsblso_bfs_numbers_TF) & (os.path.exists(f'{self.sett.data_path}/input_split_data_geometry/solkat_bsblso_geo.geojson')):
                solkat_all_geo = gpd.read_file(f'{self.sett.data_path}/input_split_data_geometry/solkat_bsblso_geo.geojson')
            else:  
                solkat_all_geo = gpd.read_file(f'{self.sett.data_path}/input_split_data_geometry/solkat_geo.geojson')
            checkpoint_to_logfile(f'import solkat_geo, {solkat_all_geo.shape[0]} rows', self.sett.log_name,  1, self.sett.show_debug_prints)    
            

            # minor transformations to str (with removing nan values)
            solkat_all_geo['DF_UID'] = solkat_all_geo['DF_UID'].astype(str)
            print('transform solkat_geo')
            
            solkat_all_pq['DF_UID'] = solkat_all_pq['DF_UID'].astype(str)    
            solkat_all_pq['SB_UUID'] = solkat_all_pq['SB_UUID'].astype(str)

            # solkat_all_pq['GWR_EGID'] = solkat_all_pq['GWR_EGID'].fillna('NAN').astype(str)
            solkat_all_pq['GWR_EGID'] = solkat_all_pq['GWR_EGID'].fillna(0).astype(int).astype(str)
            solkat_all_pq.loc[solkat_all_pq['GWR_EGID'] == '0', 'GWR_EGID'] = 'NAN'

            solkat_all_pq.rename(columns={'GWR_EGID': 'EGID'}, inplace=True)
            solkat_all_pq = solkat_all_pq.dropna(subset=['DF_UID'])
            
            solkat_all_pq['EGID_count'] = solkat_all_pq.groupby('EGID')['EGID'].transform('count')
            
            
            
            # add omitted EGIDs to SOLKAT ---------------------------------------------------------------------------------
            # old version, no EGIDs matched to solkat
            """
            if not self.sett.SOLKAT_match_missing_EGIDs_to_solkat_TF:
                solkat_v1 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(self.sett.bfs_numbers)])
                solkat_v1_wgeo = solkat_v1.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID') # merge geometry for later use
                solkat_v1_gdf = gpd.GeoDataFrame(solkat_v1_wgeo, geometry='geometry')
                solkat, solkat_gdf = copy.deepcopy(solkat_v1), copy.deepcopy(solkat_v1_gdf)
            elif self.sett.SOLKAT_match_missing_EGIDs_to_solkat_TF:
            """
            
            # the solkat df has missing EGIDs, for example row houses where the entire roof is attributed to one EGID. Attempt to 
            # 1 - add roof (perfectly overlapping roofpartitions) to solkat for all the EGIDs within the unions shape
            # 2- reduce the FLAECHE for all theses partitions by dividing it through the number of EGIDs in the union shape
            print_to_logfile('\nMatch missing EGIDs to solkat (where gwrEGIDs overlapp solkat shape but are not present as a single solkat_row)', self.sett.summary_name)
            cols_adjust_for_missEGIDs_to_solkat = self.sett.SOLKAT_cols_adjust_for_missEGIDs_to_solkat

            solkat_v2 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(self.sett.bfs_numbers)])
            solkat_v2_wgeo = solkat_v2.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID')
            solkat_v2_gdf = gpd.GeoDataFrame(solkat_v2_wgeo, geometry='geometry')
            # solkat_v2_gdf = solkat_v2_gdf[solkat_v2_gdf['EGID'] != 'NAN']
            solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == 'NAN', ['EGID', 'SB_UUID']]


            # create mapping of solkatEGIDs and missing gwrEGIDs 
            # union all shapes with the same EGID 
            solkat_union_v2EGID = solkat_v2_gdf.groupby('EGID').agg({
                'geometry': lambda x: unary_union(x),  # Combine all geometries into one MultiPolygon
                'DF_UID': lambda x: '_'.join(map(str, x))  # Concatenate DF_UID values as a single string
                }).reset_index()
            solkat_union_v2EGID = gpd.GeoDataFrame(solkat_union_v2EGID, geometry='geometry')
            

            # match gwrEGID through sjoin to solkat
            solkat_union_v2EGID = solkat_union_v2EGID.rename(columns = {'EGID': 'EGID_old_solkat'})  # rename EGID colum because gwr_EGIDs are now matched to union_shapes
            solkat_union_v2EGID.set_crs(gwr_gdf.crs, allow_override=True, inplace=True)
            join_gwr_solkat_union = gpd.sjoin(solkat_union_v2EGID, gwr_gdf, how='left')
            join_gwr_solkat_union.rename(columns = {'EGID': 'EGID_gwradded'}, inplace = True)
            checkpoint_to_logfile(f'nrows \n\tsolkat_all_pq: {solkat_all_pq.shape[0]}\t\t\tsolkat_v2_gdf: {solkat_v2_gdf.shape[0]} (remove EGID.NANs)\n\tsolkat_union_v2EGID: {solkat_union_v2EGID.shape[0]}\t\tjoin_gwr_solkat_union: {join_gwr_solkat_union.shape[0]}', self.sett.log_name, 3, self.sett.show_debug_prints)
            checkpoint_to_logfile(f'nEGID \n\tsolkat_all_pq: {solkat_all_pq["EGID"].nunique()}\t\t\tsolkat_v2_gdf: {solkat_v2_gdf["EGID"].nunique()} (remove EGID.NANs)\n\tsolkat_union_v2EGID_EGID_old: {solkat_union_v2EGID["EGID_old_solkat"].nunique()}\tjoin_gwr_solkat_union_EGID_old: {join_gwr_solkat_union["EGID_old_solkat"].nunique()}\tjoin_gwr_solkat_union_EGID_gwradded: {join_gwr_solkat_union["EGID_gwradded"].nunique()}', self.sett.log_name, 3, self.sett.show_debug_prints)


            # check EGID mapping case by case, add missing gwrEGIDs to solkat -------------------
            EGID_old_solkat_list = join_gwr_solkat_union['EGID_old_solkat'].unique()
            new_solkat_append_list = []
            add_solkat_counter, add_solkat_partition = 1, 4
            print_counter_max, i_print = 50, 0
            # n_egid, egid = 0, EGID_old_solkat_list[0]
            for n_egid, egid in enumerate(EGID_old_solkat_list):

                egid_join_union = join_gwr_solkat_union.loc[join_gwr_solkat_union['EGID_old_solkat'] == egid,]
                egid_join_union = egid_join_union.reset_index(drop = True)

                # find cases for SB_UUID proxy
                # if ('egid_proxy' in egid) & (not all(egid_join_union['EGID_gwradded'].isna())):
                #     print(f'negid {n_egid}, egid {egid}, has proxy EGID, but gwr EGIDs found in union shape, skipping proxy handling')

                if all(egid_join_union['EGID_gwradded'].isna()):  
                    # no overlapp between solkat and any GWR => skip
                    continue
                                        
                # Shapes of building that will not be included given GWR filter settings
                elif any(egid_join_union['EGID_gwradded'].isna()):  
                    solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                    # solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']

                elif all(egid_join_union['EGID_gwradded'] != np.nan): 

                    # cases
                    case1_TF = (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] == egid)
                    case2_TF = (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] != egid)
                    case3_TF = (egid_join_union.shape[0] > 1) & any(egid_join_union['EGID_gwradded'].isna())
                    case4_TF = (egid_join_union.shape[0] > 1) & (egid in egid_join_union['EGID_gwradded'].to_list())
                    case5_TF = (egid_join_union.shape[0] > 1) & (egid not in egid_join_union['EGID_gwradded'].to_list())

                    # "Best" case (unless step above applies): Shapes of building that only has 1 GWR EGID
                    if case1_TF:
                        solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                        # solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']
                        
                    # Not best case but for consistency better to keep individual solkatEGIs matches (otherwise missmatch of newer buildings with old shape partitions possible)
                    # edit: Because also roofs with GWR_EGID == NAN are considered with SB_UUID proxy, make sense to also overwrite solkat EGID for this case
                    elif case2_TF:
                        solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                        # solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']
                        solkat_subdf['EGID'] = egid_join_union['EGID_gwradded'].values[0]

                    elif case3_TF:
                        print(f'**MAJOR ERROR**: EGID {egid}, np.nan in egid_join_union[EGID_gwradded] column')

                    # Intended case: Shapes of building that has multiple GWR EGIDs within the shape boundaries
                    elif case4_TF:
                        
                        solkat_subdf_addedEGID_list = []
                        n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                        for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):
                            
                            # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                            solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                            # solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                            solkat_addedEGID['EGID'] = egid_to_add
                            
                            #extend the DF_UID with some numbers to have truely unique DF_UIDs
                            if self.sett.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                                str_suffix = str(n+1).zfill(5)
                                if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                                    solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                                elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                                    solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                            # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                            for col in cols_adjust_for_missEGIDs_to_solkat:
                                solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                            
                            solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                        
                        # concat all EGIDs within the same shape that were previously missing
                        solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)
                        
                    # Error case: Shapes of building that has multiple gwrEGIDs but does not overlap with the assigned / identical solkatEGID. 
                    # 5a (discontinued because of 5b) Not proper solution, but best for now: add matching gwrEGID to solkatEGID selection, despite the acutall gwrEGID 
                    # being placed in another shape (not necessarily though, just not in the egid_join_union)
                    # 5b edit: because nan in GWR_EGID of solkat have been replaced with SB_UUID as a proxy for coherent EGID union shapes, this case now apprears more often and
                    # must be dealt differently. Take roof shape(s), overwrite the proxyEGID (for shape union) with the gwr EGIDs within the union shape. then follow same steps 
                    # as in case4. Because EGID_old_solkat is overwritten with EGID_gwradded, no special subcase is needed wether EGID_old_solkat is a "proper" EGID or a proxyEGID
                    # using SB_UUID.
                
                    elif case5_TF:
                        # 5b case
                        solkat_subdf_addedEGID_list = []

                        n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                        for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):
                            
                            # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                            solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                            # solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                            solkat_addedEGID['EGID'] = egid_to_add
                            
                            #extend the DF_UID with some numbers to have truely unique DF_UIDs
                            if self.sett.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                                str_suffix = str(n+1).zfill(5)
                                if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                                    solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                                elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                                    solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                            # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                            for col in cols_adjust_for_missEGIDs_to_solkat:
                                solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                            
                            solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                        
                        # concat all EGIDs within the same shape that were previously missing
                        solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)
                        
                            
                            # 5a case (discontinued)
                        if False: 
                            # else: 


                            # attach a copy of one solkatEGID partition and set the EGID to the gwrEGID
                            gwrEGID_row = copy.deepcopy(egid_join_union.iloc[0])
                            # solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                            gwrEGID_row['EGID_gwradded'] = egid
                            egid_join_union = pd.concat([egid_join_union, gwrEGID_row.to_frame().T], ignore_index=True)

                            # next follow all steps as in "Intended Case" above (solkat_shape with solkatEGID and gwrEGIDs)
                            solkat_subdf_addedEGID_list = []
                            n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                            
                            for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):

                                # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                                solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                                solkat_addedEGID['EGID'] = egid_to_add
                                
                                #extend the DF_UID with some numbers to have truely unique DF_UIDs
                                if self.sett.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                                    str_suffix = str(n+1).zfill(3)
                                    if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                                        solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                                    elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                                        solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                                # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                                for col in cols_adjust_for_missEGIDs_to_solkat:
                                    solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                                
                                # shrink topology to see which partitions are affected by EGID extensions
                                # solkat_addedEGID['geometry'] =solkat_addedEGID['geometry'].buffer(-0.5, resolution=16)

                                solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                            
                            # concat all EGIDs within the same shape that were previously missing
                            solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)


                    else: 
                        if i_print < print_counter_max:
                            print(f'ERROR: EGID {egid:14}: not fitting into any case (1 to 5) for adjusting faulty SOLKAT EGIDs by matching shape to GWR_EGIDs')
                            i_print += 1
                        elif i_print == print_counter_max:
                            print(f'ERROR: EGID {egid:14}: {print_counter_max}+ ... more cases not fitting into any case (1 to 5) for adjusting faulty SOLKAT EGIDs by matching shape to GWR_EGIDs')
                            i_print += 1


                if n_egid == int(len(EGID_old_solkat_list)/add_solkat_partition):
                    checkpoint_to_logfile(f'Match gwrEGID to solkat: {add_solkat_counter}/{add_solkat_partition} partition', self.sett.log_name, 3, self.sett.show_debug_prints)
                    
                # merge all solkat partitions to new solkat df
                new_solkat_append_list.append(solkat_subdf) 

            new_solkat_gdf = gpd.GeoDataFrame(pd.concat(new_solkat_append_list, ignore_index=True), geometry='geometry')
            new_solkat = new_solkat_gdf.drop(columns = ['geometry'])
            checkpoint_to_logfile(f'Extended solkat_df by {new_solkat.shape[0] - solkat_v2_gdf.shape[0]} rows (before matching: {solkat_v2_gdf.shape[0]}, after: {new_solkat.shape[0]} rows)', self.sett.summary_name, 3, self.sett.show_debug_prints)

            solkat, solkat_gdf = copy.deepcopy(new_solkat), copy.deepcopy(new_solkat_gdf)      
            

            # SOLKAT_MONTH ====================
            # solkat_month_all_pq = pd.read_parquet(f'{self.sett.data_path}/input_split_data_geometry/solkat_month_pq.parquet')
            # checkpoint_to_logfile(f'import solkat_month_pq, {solkat_month_all_pq.shape[0]} rows,', self.sett.log_name, 1, self.sett.show_debug_prints)

            # # transformations
            # solkat_month_all_pq['SB_UUID'] = solkat_month_all_pq['SB_UUID'].astype(str)
            # solkat_month_all_pq['DF_UID'] = solkat_month_all_pq['DF_UID'].astype(str)
            # solkat_month_all_pq = solkat_month_all_pq.merge(solkat_all_pq[['DF_UID', 'BFS_NUMMER']], how = 'left', on = 'DF_UID')
            # solkat_month = solkat_month_all_pq[solkat_month_all_pq['BFS_NUMMER'].isin(self.sett.bfs_numbers)]
            
            # in polars, because RAM issues with large df in pandas
            solkat_month_all_pl = pl.read_parquet(f'{self.sett.data_path}/input_split_data_geometry/solkat_month_pq.parquet')
            checkpoint_to_logfile(f'import solkat_month_pq, {solkat_month_all_pl.shape[0]} rows,', self.sett.log_name, 1, self.sett.show_debug_prints)

            solkat_month_all_pl = solkat_month_all_pl.with_columns([
                pl.col('SB_UUID').cast(pl.Utf8),
                pl.col('DF_UID').cast(pl.Utf8)
            ])
            solkat_all_pl = pl.from_pandas(solkat_all_pq)
            solkat_month_all_pl = solkat_month_all_pl.join(solkat_all_pl.select(['DF_UID', 'BFS_NUMMER']), on='DF_UID', how='left')
            solkat_month_pl = solkat_month_all_pl.filter(pl.col('BFS_NUMMER').is_in(self.sett.bfs_numbers))
            solkat_month = solkat_month_pl.to_pandas()



            # BFS-ARE Gemeinde Type ====================
            gemeinde_type_gdf = gpd.read_file(f'{self.sett.data_path}/input/gemeindetypen_2056.gpkg/gemeindetypen_2056.gpkg', layer=None)


            # GRID_NODE ====================
            Map_egid_dsonode = pd.read_excel(f'{get_primeo_path()}/Daten_Primeo_x_UniBasel_V2.0.xlsx')
            # transformations
            Map_egid_dsonode.rename(columns={'ID_Trafostation': 'grid_node', 'Trafoleistung_kVA': 'kVA_threshold'}, inplace=True)
            Map_egid_dsonode['EGID'] = Map_egid_dsonode['EGID'].astype(str)
            Map_egid_dsonode['grid_node'] = Map_egid_dsonode['grid_node'].astype(str)

            egid_counts = Map_egid_dsonode['EGID'].value_counts()
            multip_egid_dsonode = egid_counts[egid_counts > 1].index
            single_egid_dsonode = []
            egid = multip_egid_dsonode[1]
            for egid in multip_egid_dsonode:
                subegid = Map_egid_dsonode.loc[Map_egid_dsonode['EGID'] == egid,]

                if subegid['grid_node'].nunique() == 1:
                    single_egid_dsonode.append([egid, subegid['grid_node'].iloc[0], subegid['kVA_threshold'].iloc[0]])
                elif subegid['grid_node'].nunique() > 1:
                    subegid = subegid.loc[subegid['kVA_threshold'] == subegid['kVA_threshold'].max(),]
                    single_egid_dsonode.append([egid, subegid['grid_node'].iloc[0], subegid['kVA_threshold'].iloc[0]])

            single_egid_dsonode_df = pd.DataFrame(single_egid_dsonode, columns = ['EGID', 'grid_node', 'kVA_threshold'])
            # drop duplicates and append single_egid_dsonode_df
            Map_egid_dsonode.drop(Map_egid_dsonode[Map_egid_dsonode['EGID'].isin(multip_egid_dsonode)].index, inplace = True)
            Map_egid_dsonode = pd.concat([Map_egid_dsonode, single_egid_dsonode_df], ignore_index=True)

            # create gdf of all EGIDs with grid nodes for visualization
            gwr_dsonode_gdf = gwr_all_building_gdf.merge(Map_egid_dsonode, how = 'left', on = 'EGID')
            if gwr_dsonode_gdf['grid_node'].isna().any():
                gwr_dsonode_gdf = gwr_dsonode_gdf.loc[~gwr_dsonode_gdf['grid_node'].isna()]

            # MAP: solkatdfuid > egid ---------------------------------------------------------------------------------
            Map_solkatdfuid_egid = solkat_gdf.loc[:,['DF_UID', 'DF_NUMMER', 'SB_UUID', 'EGID']].copy()
            Map_solkatdfuid_egid.rename(columns = {'GWR_EGID': 'EGID'}, inplace = True)
            Map_solkatdfuid_egid = Map_solkatdfuid_egid.loc[Map_solkatdfuid_egid['EGID'] != '']


            # MAP: egid > pv ---------------------------------------------------------------------------------
            def set_crs_to_gm_shp(gdf_CRS, gdf_a, gdf_b = None):
                gdf_a.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
                if gdf_b is not None:
                    gdf_b.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
                
                if gdf_b is None: 
                    return gdf_a
                if gdf_b is not None:
                    return gdf_a, gdf_b
                

            # find optimal buffer size ====================
            if self.sett.SOLKAT_test_loop_optim_buff_size_TF:
                print_to_logfile('\n\n Check different buffersizes!', self.sett.log_name)
                arange_start, arange_end, arange_step = self.sett.SOLKAT_test_loop_optim_buff_arang[0], self.sett.SOLKAT_test_loop_optim_buff_arang[1], self.sett.SOLKAT_test_loop_optim_buff_arang[2]
                buff_range = np.arange(arange_start, arange_end, arange_step)
                shares_xtf_duplicates = []
                for i in buff_range:# [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 2]:
                    print_to_logfile(f'buffer size: {i}', self.sett.log_name)

                    gwr_loop = copy.deepcopy(gwr_gdf)
                    gwr_loop.set_crs("EPSG:32632", allow_override=True, inplace=True)
                    gwr_loop['geometry'] = gwr_loop['geometry'].buffer(i)
                    pv_loop = copy.deepcopy(pv_gdf)
                    gwr_loop, pv_loop = set_crs_to_gm_shp(gm_shp_gdf, gwr_loop, pv_loop)
                    gwregid_pvid_loop = gpd.sjoin(pv_loop,gwr_loop, how="left", predicate="within")
                    gwregid_pvid_loop.drop(columns = ['index_right'] + [col for col in gwr_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)

                    shares = [i, 
                            round(sum(gwregid_pvid_loop['xtf_id'].isna())           /gwregid_pvid_loop['xtf_id'].nunique(),2), 
                            round(sum(gwregid_pvid_loop['xtf_id'].value_counts()==1)/gwregid_pvid_loop['xtf_id'].nunique(),2), 
                            round(sum(gwregid_pvid_loop['xtf_id'].value_counts()==2)/gwregid_pvid_loop['xtf_id'].nunique(),2), 
                            round(sum(gwregid_pvid_loop['xtf_id'].value_counts()>2) /gwregid_pvid_loop['xtf_id'].nunique(),2) ]
                    shares_xtf_duplicates.append(shares)
                    
                    print_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid_loop["EGID"].isna().sum() / gwregid_pvid_loop.shape[0] *100,2)} % of pv rows ({gwregid_pvid_loop.shape[0]}) are missing EGID', self.sett.log_name)
                    print_to_logfile(f'Duplicate shares: \tNANs\tunique\t2x\t>2x \n \t\t\t{shares[0]}\t{shares[1]}\t{shares[2]}\t{shares[3]}\t{sum(shares)}\n', self.sett.log_name)
                
                # plot shares of successful mappings
                # shares_xtf_duplicates_df = pd.DataFrame(shares_xtf_duplicates, columns = ['buffer_size', 'NANs', 'unique', '2x', '>2x'])
                # not plotted because over-exaggerated buffer is later corrected with closest neighbour matching
                # fig = px.line(shares_xtf_duplicates_df, 
                #               x='buffer_size', y=['NANs', 'unique', '2x', '>2x'],
                #               title = 'Shares of xtf_id duplicates', labels = {'buffer_size': 'Buffer Size', 'value': 'Share'}, width = 800, height = 400)
                # fig.show()
                # fig.write_html(f'{data_path_def}/output/preprep_data/by_buffersize_share_xtf_id_duplicates.html')
                checkpoint_to_logfile('buffer size optimisation finished', self.sett.log_name, 2, self.sett.show_debug_prints)


            # (continued MAP: egid > pv) ----------
            gwr_buff_gdf = copy.deepcopy(gwr_all_building_gdf)
            gwr_buff_gdf.set_crs("EPSG:32632", allow_override=True, inplace=True)
            gwr_buff_gdf['geometry'] = gwr_buff_gdf['geometry'].buffer(self.sett.SOLKAT_GWR_EGID_buffer_size)
            gwr_buff_gdf, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, gwr_buff_gdf, pv_gdf)
            checkpoint_to_logfile(f'gwr_all_building_gdf.crs == pv_gdf.crs: {gwr_buff_gdf.crs == pv_gdf.crs}', self.sett.log_name, 6, self.sett.show_debug_prints)

            gwregid_pvid_all = gpd.sjoin(pv_gdf,gwr_buff_gdf, how="left", predicate="within")
            gwregid_pvid_all.drop(columns = ['index_right'] + [col for col in gwr_all_building_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)

            # keep only unique xtf_ids 
            gwregid_pvid_unique = copy.deepcopy(gwregid_pvid_all.loc[~gwregid_pvid_all.duplicated(subset='xtf_id', keep=False)])
            xtf_duplicates =      copy.deepcopy(gwregid_pvid_all.loc[ gwregid_pvid_all.duplicated(subset='xtf_id', keep=False)])
            checkpoint_to_logfile(f'sum n_unique xtf_ids: {gwregid_pvid_unique["xtf_id"].nunique()} (unique df) +{xtf_duplicates["xtf_id"].nunique()} (duplicates df) = {gwregid_pvid_unique["xtf_id"].nunique()+xtf_duplicates["xtf_id"].nunique() }; n_unique in pv_gdf: {pv_gdf["xtf_id"].nunique()}', self.sett.log_name, 6, self.sett.show_debug_prints)
        
        
            # match duplicates with nearest neighbour
            xtf_nearestmatch_list = []
            xtf_id = xtf_duplicates['xtf_id'].unique()[0]
            for xtf_id in xtf_duplicates['xtf_id'].unique():
                gwr_sub = copy.deepcopy(gwr_buff_gdf.loc[gwr_buff_gdf['EGID'].isin(xtf_duplicates.loc[xtf_duplicates['xtf_id'] == xtf_id, 'EGID'])])
                pv_sub = copy.deepcopy(pv_gdf.loc[pv_gdf['xtf_id'] == xtf_id])
                
                assert pv_sub.crs == gwr_sub.crs
                gwr_sub['distance_to_pv'] = gwr_sub['geometry'].centroid.distance(pv_sub['geometry'].values[0])
                pv_sub['EGID'] = gwr_sub.loc[gwr_sub['distance_to_pv'].idxmin()]['EGID']

                xtf_nearestmatch_list.append(pv_sub)
            
            xtf_nearestmatches_df = pd.concat(xtf_nearestmatch_list, ignore_index=True)
            gwregid_pvid = pd.concat([gwregid_pvid_unique, xtf_nearestmatches_df], ignore_index=True).drop_duplicates()
            checkpoint_to_logfile(f'total unique xtf: {pv_gdf["xtf_id"].nunique()} (pv_gdf); {gwregid_pvid_unique["xtf_id"].nunique()+xtf_nearestmatches_df["xtf_id"].nunique()} (unique + nearest match)', self.sett.log_name, 6, self.sett.show_debug_prints)

            checkpoint_to_logfile(f'Mapping egid_pvid: {round(gwregid_pvid["EGID"].isna().sum() / gwregid_pvid.shape[0] *100,2)} % of pv rows ({gwregid_pvid.shape[0]}) are missing EGID', self.sett.log_name, 6, self.sett.show_debug_prints)
            # Map_egid_pv = gwregid_pvid.loc[gwregid_pvid['EGID'].notna(), ['EGID', 'xtf_id']].copy()
            Map_egid_pv = gwregid_pvid[['EGID', 'xtf_id']].copy()


            # CHECK SELECTION: - OMITTED SPATIAL POINTS / POLYS ---------------------------------------------------------------------------------
            print_to_logfile('\nnumber of omitted buildings because EGID is (not) / present in all of GWR / Solkat / PV / Grid_Node', self.sett.summary_name)
            print_to_logfile(f'>gwr settings: \n n bfs_numbers: {len(self.sett.bfs_numbers)}, \n year_range: {self.sett.year_range}, \n building class GKLAS: {self.sett.GWR_GKLAS}, \n building status GSTAT: {self.sett.GWR_GSTAT}, \n year of construction GBAUJ: {self.sett.GWR_GBAUJ_minmax}', self.sett.summary_name)
            omitt_gwregid_gdf = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(solkat_gdf['EGID'])])
            checkpoint_to_logfile(f'omitt_gwregid_gdf (gwr not in solkat): {omitt_gwregid_gdf.shape[0]} rows ({round((omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%), gwr[EGID].unique: {gwr_gdf["EGID"].nunique()})', self.sett.summary_name, 2, True)

            omitt_solkat_all_gwr_gdf = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_all_building_gdf['EGID'])])
            omitt_solkat_gdf = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['EGID'].isin(gwr_gdf['EGID'])])
            checkpoint_to_logfile(f'omitt_solkat_gdf (solkat not in gwr): {omitt_solkat_gdf.shape[0]} rows ({round((omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0])*100, 2)}%), solkat[EGID].unique: {solkat_gdf["EGID"].nunique()})', self.sett.summary_name, 2, True)

            omitt_pv_gdf = copy.deepcopy(pv_gdf.loc[~pv_gdf['xtf_id'].isin(gwregid_pvid['xtf_id'])])
            checkpoint_to_logfile(f'omitt_pv_gdf (pv not in gwr): {omitt_pv_gdf.shape[0]} rows ({round((omitt_pv_gdf.shape[0]/pv_gdf.shape[0])*100, 2)}%, pv[xtf_id].unique: {pv_gdf["xtf_id"].nunique()})', self.sett.summary_name, 2, True)

            omitt_gwregid_gridnode_gdf = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(Map_egid_dsonode['EGID'])])
            checkpoint_to_logfile(f'omitt_gwregid_gridnode_gdf (gwr not in gridnode): {omitt_gwregid_gridnode_gdf.shape[0]} rows ({round((omitt_gwregid_gridnode_gdf.shape[0]/gwr_gdf.shape[0])*100, 2)}%), gwr[EGID].unique: {gwr_gdf["EGID"].nunique()})', self.sett.summary_name, 2, True)

            omitt_gridnodeegid_gwr_df = copy.deepcopy(Map_egid_dsonode.loc[~Map_egid_dsonode['EGID'].isin(gwr_gdf['EGID'])])
            checkpoint_to_logfile(f'omitt_gridnodeegid_gwr_df (gridnode not in gwr): {omitt_gridnodeegid_gwr_df.shape[0]} rows ({round((omitt_gridnodeegid_gwr_df.shape[0]/Map_egid_dsonode.shape[0])*100, 2)}%), gridnode[EGID].unique: {Map_egid_dsonode["EGID"].nunique()})', self.sett.summary_name, 2, True)
            

            # CHECK SELECTION: - PRINTS TO SUMMARY LOG FILE ---------------------------------------------------------------------------------
            print_to_logfile('\n\nHow well does GWR cover other data sources', self.sett.summary_name)
            checkpoint_to_logfile(f'gwr_EGID omitted in solkat: {round(omitt_gwregid_gdf.shape[0]/gwr_gdf.shape[0]*100, 2)} %', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'solkat_EGID omitted in gwr_all_bldng: {round(omitt_solkat_all_gwr_gdf.shape[0]/solkat_gdf.shape[0]*100, 2)} %', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'solkat_EGID omitted in gwr: {round(omitt_solkat_gdf.shape[0]/solkat_gdf.shape[0]*100, 2)} %', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'pv_xtf_id omitted in gwr: {round(omitt_pv_gdf.shape[0]/pv_gdf.shape[0]*100, 2)} %', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'gwr_EGID omitted in gridnode: {round(omitt_gwregid_gridnode_gdf.shape[0]/gwr_gdf.shape[0]*100, 2)} %', self.sett.summary_name, 2, True)
            checkpoint_to_logfile(f'gridnode_EGID omitted in gwr: {round(omitt_gridnodeegid_gwr_df.shape[0]/Map_egid_dsonode.shape[0]*100, 2)} %', self.sett.summary_name, 2, True)


            # EXPORTS (parquet) ---------------------------------------------------------------------------------
            df_to_export_names  = ['pv', 'solkat', 'solkat_month', 'Map_egid_dsonode', 'Map_solkatdfuid_egid', 'Map_egid_pv', ]
            df_to_export_list   = [ pv,   solkat,   solkat_month,   Map_egid_dsonode,   Map_solkatdfuid_egid,   Map_egid_pv , ]
            for i, df in enumerate(df_to_export_list):
                df.to_parquet(f'{self.sett.preprep_path}/{df_to_export_names[i]}.parquet')
                # df.to_csv(f'{self.sett.preprep_path}/{df_to_export_names[i]}.csv', sep=';', index=False)
                checkpoint_to_logfile(f'{df_to_export_names[i]} exported to prepreped data', self.sett.log_name, 1, self.sett.show_debug_prints)


            # EXPORT SPATIAL DATA ---------------------------------------------------------------------------------
            gdf_to_export_names = [ 'gm_shp_gdf', 'pv_gdf', 'solkat_gdf', 'gwr_gdf','gwr_buff_gdf', 'gwr_all_building_gdf', 
                                    'omitt_gwregid_gdf', 'omitt_solkat_gdf', 'omitt_pv_gdf', 'omitt_gwregid_gridnode_gdf', 
                                    'gwr_dsonode_gdf', 'gemeinde_type_gdf', ]
            gdf_to_export_list = [  gm_shp_gdf, pv_gdf, solkat_gdf, gwr_gdf, gwr_buff_gdf, gwr_all_building_gdf, 
                                    omitt_gwregid_gdf, omitt_solkat_gdf, omitt_pv_gdf, omitt_gwregid_gridnode_gdf, 
                                    gwr_dsonode_gdf, gemeinde_type_gdf, ]
            
            for i,g in enumerate(gdf_to_export_list):
                cols_DATUM = [col for col in g.columns if 'DATUM' in col]
                g.drop(columns = cols_DATUM, inplace = True)
                # for each gdf export needs to be adjusted so it is carried over into the geojson file.
                g.set_crs("EPSG:2056", allow_override = True, inplace = True)   

                print_to_logfile(f'CRS for {gdf_to_export_names[i]}: {g.crs}', self.sett.log_name)
                checkpoint_to_logfile(f'exported {gdf_to_export_names[i]}', self.sett.log_name , 4, self.sett.show_debug_prints)

                with open(f'{self.sett.preprep_path}/{gdf_to_export_names[i]}.geojson', 'w') as f:
                    f.write(g.to_json()) 


        def preprep_data_import_ts_AND_match_households(self):
            """
            Input: 
                - Input: DataAggScenario_Settings
            Tasks: 
                - 1) Import demand time series data and aggregate it to 4 demand archetypes.
                - 2) Match the time series to the households IDs dependent on building characteristics (e.g. flat/house size, electric heating, etc.)
            Output to preprep dir: 
                - Export all the mappings and time series data.
            """
            # SETUP --------------------------------------
            print_to_logfile('run function: import_demand_TS_AND_match_households.py', self.sett.log_name)


            # IMPORT CONSUMER DATA -----------------------------------------------------------------
            print_to_logfile(f'\nIMPORT CONSUMER DATA {10*"*"}', self.sett.log_name) 
            

            # DEMAND DATA SOURCE: NETFLEX ============================================================
            # if self.sett.DEMAND_input_data_source == "NETFLEX" :
            if False:
                # import demand TS --------
                netflex_consumers_list = glob.glob(f'{self.sett.data_path}/input/NETFLEX_consumers/ID*') # os.listdir(f'{self.sett.data_path}/input/NETFLEX_consumers')
                        
                all_assets_list = []
                # c = netflex_consumers_list[1]
                for path in netflex_consumers_list:
                    f = open(path)
                    data = json.load(f)
                    assets = data['assets']['list'] 
                    all_assets_list.extend(assets)
                
                without_id = [a.split('_ID')[0] for a in all_assets_list]
                all_assets_unique = list(set(without_id))
                checkpoint_to_logfile(f'consumer demand TS contains assets: {all_assets_unique}', self.sett.log_name, 2, self.sett.show_debug_prints)

                # aggregate demand for each consumer
                agg_demand_df = pd.DataFrame()
                # netflex_consumers_list = netflex_consumers_list if not smaller_import_def else netflex_consumers_list[0:40]

                # for c, c_n in enumerate() netflex_consumers_list:
                for i_path, path in enumerate(netflex_consumers_list):
                    c_demand_id, c_demand_tech, c_demand_asset, c_demand_t, c_demand_values = [], [], [], [], []

                    f = open(path)
                    data = json.load(f)
                    assets = data['assets']['list'] 

                    a = assets[0]
                    for a in assets:
                        if 'asset_time_series' in data['assets'][a].keys():
                            demand = data['assets'][a]['asset_time_series']
                            c_demand_id.extend([f'ID{a.split("_ID")[1]}']*len(demand))
                            c_demand_tech.extend([a.split('_ID')[0]]*len(demand))
                            c_demand_asset.extend([a]*len(demand))
                            c_demand_t.extend(demand.keys())
                            c_demand_values.extend(demand.values())

                    c_demand_df = pd.DataFrame({'id': c_demand_id, 'tech': c_demand_tech, 'asset': c_demand_asset, 't': c_demand_t, 'value': c_demand_values})
                    agg_demand_df = pd.concat([agg_demand_df, c_demand_df])
                    
                    if (i_path + 1) % (len(netflex_consumers_list) // 4) == 0:
                        id_name = f'ID{path.split("ID")[-1].split(".json")[0]}'
                        checkpoint_to_logfile(f'exported demand TS for consumer {id_name}, {i_path+1} of {len(netflex_consumers_list)}', self.sett.log_name, 2, self.sett.show_debug_prints)
                
                # remove pv assets because they also have negative values
                agg_demand_df = agg_demand_df[agg_demand_df['tech'] != 'pv']

                agg_demand_df['value'] = agg_demand_df['value'] * 1000 # it appears that values are calculated in MWh, need kWh

                # plot TS for certain consumers by assets
                plot_ids =['ID100', 'ID101', 'ID102', ]
                plot_df = agg_demand_df[agg_demand_df['id'].isin(plot_ids)]
                fig = px.line(plot_df, x='t', y='value', color='asset', title='Demand TS for selected consumers')
                # fig.show()

                # export aggregated demand for all NETFLEX consumer assets
                agg_demand_df.to_parquet(f'{self.sett.preprep_path}/demand_ts.parquet')
                checkpoint_to_logfile('exported demand TS for all consumers', self.sett.log_name, self.sett.show_debug_prints)
                

                # AGGREGATE DEMAND TYPES -----------------------------------------------------------------
                # aggregate demand TS for defined consumer types 
                # demand upper/lower 50 percentile, with/without heat pump
                # get IDs for each subcatergory
                print_to_logfile(f'\nAGGREGATE DEMAND TYPES {10*"*"}', self.sett.log_name)
                def get_IDs_upper_lower_totalconsumpttion_by_hp(df, hp_TF = True,  up_low50percent = "upper"):
                    id_with_hp = df[df['tech'] == 'hp']['id'].unique()
                    if hp_TF: 
                        filtered_df = df[df['id'].isin(id_with_hp)]
                    elif not hp_TF:
                        filtered_df = df[~df['id'].isin(id_with_hp)]

                    filtered_df = filtered_df.loc[filtered_df['tech'] != 'pv']

                    total_consumption = filtered_df.groupby('id')['value'].sum().reset_index()
                    mean_value = total_consumption['value'].mean()
                    id_upper_half = total_consumption.loc[total_consumption['value'] > mean_value, 'id']
                    id_lower_half = total_consumption.loc[total_consumption['value'] < mean_value, 'id']

                    if up_low50percent == "upper":
                        return id_upper_half
                    elif up_low50percent == "lower":
                        return id_lower_half
                
                # classify consumers to later aggregate them into demand types
                ids_high_wiHP = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = True, up_low50percent = "upper")
                ids_low_wiHP  = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = True, up_low50percent = "lower")
                ids_high_noHP = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = False, up_low50percent = "upper")
                ids_low_noHP  = get_IDs_upper_lower_totalconsumpttion_by_hp(agg_demand_df, hp_TF = False, up_low50percent = "lower")

                # aggregate demand types
                demandtypes = pd.DataFrame()
                t_sequence = agg_demand_df['t'].unique()

                demandtypes['t'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().keys()
                # demandtypes['high_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().values
                # demandtypes['low_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_wiHP)].groupby('t')['value'].mean().values
                # demandtypes['high_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_noHP)].groupby('t')['value'].mean().values
                # demandtypes['low_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_noHP)].groupby('t')['value'].mean().values
                demandtypes['high_DEMANDprox_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_wiHP)].groupby('t')['value'].mean().values
                demandtypes['low_DEMANDprox_wiHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_wiHP)].groupby('t')['value'].mean().values
                demandtypes['high_DEMANDprox_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_high_noHP)].groupby('t')['value'].mean().values
                demandtypes['low_DEMANDprox_noHP'] = agg_demand_df.loc[agg_demand_df['id'].isin(ids_low_noHP)].groupby('t')['value'].mean().values

                demandtypes['t'] = pd.Categorical(demandtypes['t'], categories=t_sequence, ordered=True)
                demandtypes = demandtypes.sort_values(by = 't')
                demandtypes = demandtypes.reset_index(drop=True)

                demandtypes.to_parquet(f'{self.sett.preprep_path}/demandtypes.parquet')
                demandtypes.to_csv(f'{self.sett.preprep_path}/demandtypes.csv', sep=';', index=False)
                checkpoint_to_logfile('exported demand types', self.sett.log_name, self.sett.show_debug_prints)

                # plot demand types with plotly
                fig = px.line(demandtypes, x='t', y=['high_DEMANDprox_wiHP', 'low_DEMANDprox_wiHP', 'high_DEMANDprox_noHP', 'low_DEMANDprox_noHP'], title='Demand types')
                # fig.show()
                fig.write_html(f'{self.sett.preprep_path}/demandtypes.html')
                demandtypes['high_DEMANDprox_wiHP'].sum(), demandtypes['low_DEMANDprox_wiHP'].sum(), demandtypes['high_DEMANDprox_noHP'].sum(), demandtypes['low_DEMANDprox_noHP'].sum()


                # MATCH DEMAND TYPES TO HOUSEHOLDS -----------------------------------------------------------------
                print_to_logfile(f'\nMATCH DEMAND TYPES TO HOUSEHOLDS {10*"*"}', self.sett.log_name)

                # import GWR and PV --------
                gwr_all = pd.read_parquet(f'{self.sett.preprep_path}/gwr.parquet')
                checkpoint_to_logfile('imported gwr data', self.sett.log_name, self.sett.show_debug_prints)
                
                # transformations
                gwr_all[self.sett.GWR_DEMAND_proxy] = pd.to_numeric(gwr_all[self.sett.GWR_DEMAND_proxy], errors='coerce')
                gwr_all['GBAUJ'] = pd.to_numeric(gwr_all['GBAUJ'], errors='coerce')
                gwr_all.dropna(subset = ['GBAUJ'], inplace = True)
                gwr_all['GBAUJ'] = gwr_all['GBAUJ'].astype(int)

                # selection based on GWR specifications -------- 
                # select columns GSTAT that are within list ['1110','1112'] and GKLAS in ['1234','2345']
                gwr = gwr_all[(gwr_all['GSTAT'].isin(self.sett.GWR_GSTAT)) & 
                            (gwr_all['GKLAS'].isin(self.sett.GWR_GKLAS)) & 
                            (gwr_all['GBAUJ'] >= self.sett.GWR_GBAUJ_minmax[0]) &
                            (gwr_all['GBAUJ'] <= self.sett.GWR_GBAUJ_minmax[1])]
                checkpoint_to_logfile(f'filtered vs unfiltered gwr: shape ({gwr.shape[0]} vs {gwr_all.shape[0]}), EGID.nunique ({gwr["EGID"].nunique()} vs {gwr_all ["EGID"].nunique()})', self.sett.log_name, 2, self.sett.show_debug_prints)
                
                def get_IDs_upper_lower_DEMAND_by_hp(df, DEMAND_col = self.sett.GWR_DEMAND_proxy,  hp_TF = True,  up_low50percent = "upper"):
                    id_with_hp = df[df['GWAERZH1'].isin(self.sett.GWR_GWAERZH)]['EGID'].unique()
                    if hp_TF: 
                        filtered_df = df[df['EGID'].isin(id_with_hp)]
                    elif not hp_TF:
                        filtered_df = df[~df['EGID'].isin(id_with_hp)]
                    
                    mean_value = filtered_df[DEMAND_col].mean()
                    id_upper_half = filtered_df.loc[filtered_df[DEMAND_col] > mean_value, 'EGID']
                    id_lower_half = filtered_df.loc[filtered_df[DEMAND_col] < mean_value, 'EGID']
                    if up_low50percent == "upper":
                        return id_upper_half.tolist()
                    elif up_low50percent == "lower":
                        return id_lower_half.tolist()
                    
                high_DEMANDprox_wiHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = True, up_low50percent = "upper")
                low_DEMANDprox_wiHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = True, up_low50percent = "lower")
                high_DEMANDprox_noHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = False, up_low50percent = "upper")
                low_DEMANDprox_noHP_list = get_IDs_upper_lower_DEMAND_by_hp(gwr, hp_TF = False, up_low50percent = "lower")


                # sanity check --------
                print_to_logfile('sanity check gwr classifications', self.sett.log_name)
                gwr_classified_list = [high_DEMANDprox_wiHP_list, low_DEMANDprox_wiHP_list, high_DEMANDprox_noHP_list, low_DEMANDprox_noHP_list]
                gwr_classified_names= ['high_DEMANDprox_wiHP_list', 'low_DEMANDprox_wiHP_list', 'high_DEMANDprox_noHP_list', 'low_DEMANDprox_noHP_list']

                for chosen_lst_idx, chosen_list in enumerate(gwr_classified_list):
                    chosen_set = set(chosen_list)

                    for i, lst in enumerate(gwr_classified_list):
                        if i != chosen_lst_idx:
                            other_set = set(lst)
                            common_ids = chosen_set.intersection(other_set)
                            print_to_logfile(f"No. of common IDs between {gwr_classified_names[chosen_lst_idx]} and {gwr_classified_names[i]}: {len(common_ids)}", self.sett.log_name)
                    print_to_logfile('\n', self.sett.log_name)

                # precent of classified buildings
                n_classified = sum([len(lst) for lst in gwr_classified_list])
                n_all = len(gwr['EGID'])
                print_to_logfile(f'{n_classified} of {n_all} ({round(n_classified/n_all*100, 2)}%) gwr rows are classfied', self.sett.log_name)
                

                # export to JSON --------
                Map_demandtype_EGID ={
                    'high_DEMANDprox_wiHP': high_DEMANDprox_wiHP_list,
                    'low_DEMANDprox_wiHP': low_DEMANDprox_wiHP_list,
                    'high_DEMANDprox_noHP': high_DEMANDprox_noHP_list,
                    'low_DEMANDprox_noHP': low_DEMANDprox_noHP_list,
                }
                with open(f'{self.sett.preprep_path}/Map_demandtype_EGID.json', 'w') as f:
                    json.dump(Map_demandtype_EGID, f)
                checkpoint_to_logfile('exported Map_demandtype_EGID.json', self.sett.log_name, self.sett.show_debug_prints)

                Map_EGID_demandtypes = {}
                for type, egid_list in Map_demandtype_EGID.items():
                    for egid in egid_list:
                        Map_EGID_demandtypes[egid] = type
                with open(f'{self.sett.preprep_path}/Map_EGID_demandtypes.json', 'w') as f:
                    json.dump(Map_EGID_demandtypes, f)
                checkpoint_to_logfile('exported Map_EGID_demandtypes.json', self.sett.log_name, self.sett.show_debug_prints)


            # DEMAND DATA SOURCE: SwissStore ============================================================
            elif self.sett.DEMAND_input_data_source == "SwissStore" :
                swstore_arch_typ_factors  = pd.read_excel(f'{self.sett.data_path}/input/SwissStore_DemandData/12.swisstore_table12_unige.xlsx', sheet_name='Feuil1')
                swstore_arch_typ_master   = pd.read_csv(f'{self.sett.data_path}/input/SwissStore_DemandData/Master_table_archetype.csv', sep=';')
                swstore_sfhmfh_ts         = pd.read_excel(f'{self.sett.data_path}/input/SwissStore_DemandData/Electricity_demand_SFH_MFH.xlsx', sheet_name='dmnd_prof_sfh_mfh_avg')
                gwr                       = pd.read_parquet(f'{self.sett.preprep_path}/gwr.parquet')
                gwr_all_building_gdf      = gpd.read_file(f'{self.sett.preprep_path}/gwr_all_building_gdf.geojson')
                gemeinde_type_gdf         = gpd.read_file(f'{self.sett.preprep_path}/gemeinde_type_gdf.geojson')


                # classify EGIDs into SFH / MFH, Rural / Urban -------------------------------------------------

                # get ARE type classification
                gwr_all_building_gdf['ARE_typ'] = ''
                gwr_all_building_gdf = gpd.sjoin(gwr_all_building_gdf, gemeinde_type_gdf[['NAME', 'TYP', 'BFS_NO', 'geometry']], 
                                                 how='left', predicate='intersects')
                gwr_all_building_gdf.rename(columns={'NAME': 'ARE_NAME', 'TYP': 'ARE_TYP', }, inplace=True)
                for k,v in self.sett.GWR_AREtypology.items():
                    gwr_all_building_gdf.loc[gwr_all_building_gdf['ARE_TYP'].isin(v), 'ARE_typ'] = k

                # get SFH / MFH classification from GWR data
                gwr_all_building_gdf['sfhmfh_typ'] = ''
                for k,v in self.sett.GWR_SFHMFHtypology.items():
                    gwr_all_building_gdf.loc[gwr_all_building_gdf['GKLAS'].isin(v), 'sfhmfh_typ'] = k
                gwr_all_building_gdf.loc[gwr_all_building_gdf['sfhmfh_typ'] == '', 'sfhmfh_typ'] = self.sett.GWR_SFHMFH_outsample_proxy

                # build swstore_type to attach swstore factors
                gwr_all_building_gdf['arch_typ'] = gwr_all_building_gdf['sfhmfh_typ'].str.cat(gwr_all_building_gdf['ARE_typ'], sep='-')
                gwr_all_building_gdf = gwr_all_building_gdf.merge(swstore_arch_typ_factors[['arch_typ', 'elec_dem_ind_cecb', ]])
                gwr_all_building_gdf.rename(columns={'elec_dem_ind_cecb': 'demand_elec_pGAREA'}, inplace=True)

                # attach information to gwr and export
                gwr = gwr.merge(gwr_all_building_gdf[['EGID', 'ARE_typ', 'sfhmfh_typ', 'arch_typ', 'demand_elec_pGAREA']], on='EGID', how='left')
                gwr_all_building_df = gwr_all_building_gdf.drop(columns=['geometry', ]).copy()

                # export 
                gwr.to_parquet(f'{self.sett.preprep_path}/gwr.parquet')
                gwr.to_csv(f'{self.sett.preprep_path}/gwr.csv', sep=';', index=False)  

                gwr_all_building_df.to_parquet(f'{self.sett.preprep_path}/gwr_all_building_df.parquet')
                gwr_all_building_df.to_csv(f'{self.sett.preprep_path}/gwr_all_building_df.csv', sep=';', index=False)

                gwr_all_building_gdf.to_file(f'{self.sett.preprep_path}/gwr_all_building_gdf.geojson', driver='GeoJSON')


                # transform demand profiles to TS -------------------------------------------------
                swstore_sfhmfh_ts = swstore_sfhmfh_ts.dropna(subset=['MFH', 'SFH'], how='all')
                swstore_sfhmfh_ts['t'] = [f't_{i+1}' for i in range(len(swstore_sfhmfh_ts))]
                swstore_sfhmfh_ts['t_int'] = [i+1 for i in range(len(swstore_sfhmfh_ts))]
                demandtypes_ts = copy.deepcopy(swstore_sfhmfh_ts)

                # export
                demandtypes_ts.to_parquet(f'{self.sett.preprep_path}/demandtypes_ts.parquet')
                demandtypes_ts.to_csv(f'{self.sett.preprep_path}/demandtypes_ts.csv',)
                swstore_arch_typ_factors.to_parquet(f'{self.sett.preprep_path}/swstore_arch_typ_factors.parquet')
                swstore_arch_typ_factors.to_csv(f'{self.sett.preprep_path}/swstore_arch_typ_factors.csv')
                swstore_arch_typ_master.to_parquet(f'{self.sett.preprep_path}/swstore_arch_typ_master.parquet')
                swstore_arch_typ_master.to_csv(f'{self.sett.preprep_path}/swstore_arch_typ_master.csv', sep=';')



        def preprep_data_import_meteo_data(self):
            """
            Input: 
                - Input: DataAggScenario_Settings
            Tasks:
                - Import meteo data from a source, select only the relevant time frame store data to prepreped data folder
                - Contains all the relevant radiation and temperature data to calculate the production potential for a given roof
            Output to preprep dir:
                - meteo.parquet
            """
            
            # SETUP --------------------------------------
            print_to_logfile('run function: import_demand_TS_AND_match_households.py', self.sett.log_name)


            # IMPORT METEO DATA ============================================================================
            print_to_logfile(f'\nIMPORT METEO DATA {10*"*"}', self.sett.log_name)

            # import meteo data --------
            meteo = pd.read_csv(f'{self.sett.data_path}/input/Meteoblue_BSBL/Meteodaten_Basel_2018_2024_reduziert_bereinigt.csv')

            # transformations
            meteo['timestamp'] = pd.to_datetime(meteo['timestamp'], format = '%d.%m.%Y %H:%M:%S')

            # select relevant time frame
            start_stamp = pd.to_datetime(f'01.01.{self.sett.year_range[0]}', format = '%d.%m.%Y')
            end_stamp = pd.to_datetime(f'31.12.{self.sett.year_range[1]}', format = '%d.%m.%Y')
            meteo = meteo[(meteo['timestamp'] >= start_stamp) & (meteo['timestamp'] <= end_stamp)]
            
            # export --------
            meteo.to_parquet(f'{self.sett.preprep_path}/meteo.parquet')
            checkpoint_to_logfile('exported meteo data', self.sett.log_name, self.sett.show_debug_prints)

            # MATCH WEATHER STATIONS TO HOUSEHOLDS ============================================================================


        def extend_data_get_angle_tilt_table(self):
            """
            Input:
                - DataAggScenario_Settings
            Tasks: 
                - Create a table with the angles and tilts to find a reduction factor for a roof area with a certain
                  angle and tilt. Using the data from https://echtsolar.de/photovoltaik-neigungswinkel/, I transform the 
                  copy-pasted tuples to a pandas dataframe.
            Output to preprep dir:
                - angle_tilt_table.parquet
            """

            # SETUP -------------------
            print_to_logfile('run function: get_angle_tilt_table', self.sett.log_name)

            # SOURCE: table was retreived from this site: https://echtsolar.de/photovoltaik-neigungswinkel/
            # date 29.08.24
            
            # IMPORT DF -------------------
            index_angle = [-180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
            index_tilt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
            tuples_iter = list(itertools.product(index_angle, index_tilt))

            tuples = [(-180, 0), (-180, 5), (-180, 10), (-180, 15), (-180, 20), (-180, 25), (-180, 30), (-180, 35), (-180, 40), (-180, 45), (-180, 50), (-180, 55), (-180, 60), (-180, 65), (-180, 70), (-180, 75), (-180, 80), (-180, 85), (-180, 90), 
                    (-170, 0), (-170, 5), (-170, 10), (-170, 15), (-170, 20), (-170, 25), (-170, 30), (-170, 35), (-170, 40), (-170, 45), (-170, 50), (-170, 55), (-170, 60), (-170, 65), (-170, 70), (-170, 75), (-170, 80), (-170, 85), (-170, 90), 
                    (-160, 0), (-160, 5), (-160, 10), (-160, 15), (-160, 20), (-160, 25), (-160, 30), (-160, 35), (-160, 40), (-160, 45), (-160, 50), (-160, 55), (-160, 60), (-160, 65), (-160, 70), (-160, 75), (-160, 80), (-160, 85), (-160, 90), 
                    (-150, 0), (-150, 5), (-150, 10), (-150, 15), (-150, 20), (-150, 25), (-150, 30), (-150, 35), (-150, 40), (-150, 45), (-150, 50), (-150, 55), (-150, 60), (-150, 65), (-150, 70), (-150, 75), (-150, 80), (-150, 85), (-150, 90), 
                    (-140, 0), (-140, 5), (-140, 10), (-140, 15), (-140, 20), (-140, 25), (-140, 30), (-140, 35), (-140, 40), (-140, 45), (-140, 50), (-140, 55), (-140, 60), (-140, 65), (-140, 70), (-140, 75), (-140, 80), (-140, 85), (-140, 90),
                    (-130, 0), (-130, 5), (-130, 10), (-130, 15), (-130, 20), (-130, 25), (-130, 30), (-130, 35), (-130, 40), (-130, 45), (-130, 50), (-130, 55), (-130, 60), (-130, 65), (-130, 70), (-130, 75), (-130, 80), (-130, 85), (-130, 90),
                    (-120, 0), (-120, 5), (-120, 10), (-120, 15), (-120, 20), (-120, 25), (-120, 30), (-120, 35), (-120, 40), (-120, 45), (-120, 50), (-120, 55), (-120, 60), (-120, 65), (-120, 70), (-120, 75), (-120, 80), (-120, 85), (-120, 90),
                    (-110, 0), (-110, 5), (-110, 10), (-110, 15), (-110, 20), (-110, 25), (-110, 30), (-110, 35), (-110, 40), (-110, 45), (-110, 50), (-110, 55), (-110, 60), (-110, 65), (-110, 70), (-110, 75), (-110, 80), (-110, 85), (-110, 90),
                    (-100, 0), (-100, 5), (-100, 10), (-100, 15), (-100, 20), (-100, 25), (-100, 30), (-100, 35), (-100, 40), (-100, 45), (-100, 50), (-100, 55), (-100, 60), (-100, 65), (-100, 70), (-100, 75), (-100, 80), (-100, 85), (-100, 90),
                    (-90, 0), (-90, 5), (-90, 10), (-90, 15), (-90, 20), (-90, 25), (-90, 30), (-90, 35), (-90, 40), (-90, 45), (-90, 50), (-90, 55), (-90, 60), (-90, 65), (-90, 70), (-90, 75), (-90, 80), (-90, 85), (-90, 90),
                    (-80, 0), (-80, 5), (-80, 10), (-80, 15), (-80, 20), (-80, 25), (-80, 30), (-80, 35), (-80, 40), (-80, 45), (-80, 50), (-80, 55), (-80, 60), (-80, 65), (-80, 70), (-80, 75), (-80, 80), (-80, 85), (-80, 90),
                    (-70, 0), (-70, 5), (-70, 10), (-70, 15), (-70, 20), (-70, 25), (-70, 30), (-70, 35), (-70, 40), (-70, 45), (-70, 50), (-70, 55), (-70, 60), (-70, 65), (-70, 70), (-70, 75), (-70, 80), (-70, 85), (-70, 90),
                    (-60, 0), (-60, 5), (-60, 10), (-60, 15), (-60, 20), (-60, 25), (-60, 30), (-60, 35), (-60, 40), (-60, 45), (-60, 50), (-60, 55), (-60, 60), (-60, 65), (-60, 70), (-60, 75), (-60, 80), (-60, 85), (-60, 90),
                    (-50, 0), (-50, 5), (-50, 10), (-50, 15), (-50, 20), (-50, 25), (-50, 30), (-50, 35), (-50, 40), (-50, 45), (-50, 50), (-50, 55), (-50, 60), (-50, 65), (-50, 70), (-50, 75), (-50, 80), (-50, 85), (-50, 90),
                    (-40, 0), (-40, 5), (-40, 10), (-40, 15), (-40, 20), (-40, 25), (-40, 30), (-40, 35), (-40, 40), (-40, 45), (-40, 50), (-40, 55), (-40, 60), (-40, 65), (-40, 70), (-40, 75), (-40, 80), (-40, 85), (-40, 90),
                    (-30, 0), (-30, 5), (-30, 10), (-30, 15), (-30, 20), (-30, 25), (-30, 30), (-30, 35), (-30, 40), (-30, 45), (-30, 50), (-30, 55), (-30, 60), (-30, 65), (-30, 70), (-30, 75), (-30, 80), (-30, 85), (-30, 90),
                    (-20, 0), (-20, 5), (-20, 10), (-20, 15), (-20, 20), (-20, 25), (-20, 30), (-20, 35), (-20, 40), (-20, 45), (-20, 50), (-20, 55), (-20, 60), (-20, 65), (-20, 70), (-20, 75), (-20, 80), (-20, 85), (-20, 90),
                    (-10, 0), (-10, 5), (-10, 10), (-10, 15), (-10, 20), (-10, 25), (-10, 30), (-10, 35), (-10, 40), (-10, 45), (-10, 50), (-10, 55), (-10, 60), (-10, 65), (-10, 70), (-10, 75), (-10, 80), (-10, 85), (-10, 90),
                    (0, 0), (0, 5), (0, 10), (0, 15), (0, 20), (0, 25), (0, 30), (0, 35), (0, 40), (0, 45), (0, 50), (0, 55), (0, 60), (0, 65), (0, 70), (0, 75), (0, 80), (0, 85), (0, 90),
                    (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55), (10, 60), (10, 65), (10, 70), (10, 75), (10, 80), (10, 85), (10, 90),
                    (20, 0), (20, 5), (20, 10), (20, 15), (20, 20), (20, 25), (20, 30), (20, 35), (20, 40), (20, 45), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (20, 75), (20, 80), (20, 85), (20, 90),
                    (30, 0), (30, 5), (30, 10), (30, 15), (30, 20), (30, 25), (30, 30), (30, 35), (30, 40), (30, 45), (30, 50), (30, 55), (30, 60), (30, 65), (30, 70), (30, 75), (30, 80), (30, 85), (30, 90),
                    (40, 0), (40, 5), (40, 10), (40, 15), (40, 20), (40, 25), (40, 30), (40, 35), (40, 40), (40, 45), (40, 50), (40, 55), (40, 60), (40, 65), (40, 70), (40, 75), (40, 80), (40, 85), (40, 90),
                    (50, 0), (50, 5), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30), (50, 35), (50, 40), (50, 45), (50, 50), (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90),
                    (60, 0), (60, 5), (60, 10), (60, 15), (60, 20), (60, 25), (60, 30), (60, 35), (60, 40), (60, 45), (60, 50), (60, 55), (60, 60), (60, 65), (60, 70), (60, 75), (60, 80), (60, 85), (60, 90),
                    (70, 0), (70, 5), (70, 10), (70, 15), (70, 20), (70, 25), (70, 30), (70, 35), (70, 40), (70, 45), (70, 50), (70, 55), (70, 60), (70, 65), (70, 70), (70, 75), (70, 80), (70, 85), (70, 90),
                    (80, 0), (80, 5), (80, 10), (80, 15), (80, 20), (80, 25), (80, 30), (80, 35), (80, 40), (80, 45), (80, 50), (80, 55), (80, 60), (80, 65), (80, 70), (80, 75), (80, 80), (80, 85), (80, 90),
                    (90, 0), (90, 5), (90, 10), (90, 15), (90, 20), (90, 25), (90, 30), (90, 35), (90, 40), (90, 45), (90, 50), (90, 55), (90, 60), (90, 65), (90, 70), (90, 75), (90, 80), (90, 85), (90, 90),
                    (100, 0), (100, 5), (100, 10), (100, 15), (100, 20), (100, 25), (100, 30), (100, 35), (100, 40), (100, 45), (100, 50), (100, 55), (100, 60), (100, 65), (100, 70), (100, 75), (100, 80), (100, 85), (100, 90),
                    (110, 0), (110, 5), (110, 10), (110, 15), (110, 20), (110, 25), (110, 30), (110, 35), (110, 40), (110, 45), (110, 50), (110, 55), (110, 60), (110, 65), (110, 70), (110, 75), (110, 80), (110, 85), (110, 90),
                    (120, 0), (120, 5), (120, 10), (120, 15), (120, 20), (120, 25), (120, 30), (120, 35), (120, 40), (120, 45), (120, 50), (120, 55), (120, 60), (120, 65), (120, 70), (120, 75), (120, 80), (120, 85), (120, 90),
                    (130, 0), (130, 5), (130, 10), (130, 15), (130, 20), (130, 25), (130, 30), (130, 35), (130, 40), (130, 45), (130, 50), (130, 55), (130, 60), (130, 65), (130, 70), (130, 75), (130, 80), (130, 85), (130, 90),
                    (140, 0), (140, 5), (140, 10), (140, 15), (140, 20), (140, 25), (140, 30), (140, 35), (140, 40), (140, 45), (140, 50), (140, 55), (140, 60), (140, 65), (140, 70), (140, 75), (140, 80), (140, 85), (140, 90),
                    (150, 0), (150, 5), (150, 10), (150, 15), (150, 20), (150, 25), (150, 30), (150, 35), (150, 40), (150, 45), (150, 50), (150, 55), (150, 60), (150, 65), (150, 70), (150, 75), (150, 80), (150, 85), (150, 90),
                    (160, 0), (160, 5), (160, 10), (160, 15), (160, 20), (160, 25), (160, 30), (160, 35), (160, 40), (160, 45), (160, 50), (160, 55), (160, 60), (160, 65), (160, 70), (160, 75), (160, 80), (160, 85), (160, 90),
                    (170, 0), (170, 5), (170, 10), (170, 15), (170, 20), (170, 25), (170, 30), (170, 35), (170, 40), (170, 45), (170, 50), (170, 55), (170, 60), (170, 65), (170, 70), (170, 75), (170, 80), (170, 85), (170, 90),
                    (180, 0), (180, 5), (180, 10), (180, 15), (180, 20), (180, 25), (180, 30), (180, 35), (180, 40), (180, 45), (180, 50), (180, 55), (180, 60), (180, 65), (180, 70), (180, 75), (180, 80), (180, 85), (180, 90)
                    ]
            index = pd.MultiIndex.from_tuples(tuples, names=['angle', 'tilt'])

            values = [89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2, 
                    89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
                    89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
                    89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
                    89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
                    89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
                    89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
                    89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
                    89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
                    89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
                    89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
                    89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
                    89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
                    89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
                    89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
                    89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
                    89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
                    89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
                    89.0, 92.1, 94.7, 96.8, 98.4, 99.5, 100,  100 , 99.5, 98.3, 96.8, 94.8, 92.3, 89.3, 85.8, 81.9, 77.6, 73.1, 68.1,
                    89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
                    89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
                    89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
                    89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
                    89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
                    89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
                    89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
                    89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
                    89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
                    89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
                    89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
                    89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
                    89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
                    89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
                    89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
                    89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
                    89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
                    89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2
                    ] 
            
            angle_tilt_df = pd.DataFrame(data = values, index = index, columns = ['efficiency_factor'])
            angle_tilt_df['efficiency_factor'] = angle_tilt_df['efficiency_factor'] / 100

            # export df ----------
            angle_tilt_df.to_parquet(f'{self.sett.preprep_path}/angle_tilt_df.parquet')
            angle_tilt_df.to_csv(f'{self.sett.preprep_path}/angle_tilt_df.csv')
            # angle_tilt_df.to_parquet(f'{data_path_def}/output/{name_dir_import_def}/angle_tilt_df.parquet')
            # return angle_tilt_df





# ======================================================================================================
# RUN SCENARIOS
# ======================================================================================================
if __name__ == '__main__':
    dataagg_scen_list = [

        DataAggScenario_Settings(
            name_dir_export = 'preprep_debug',
            # kt_numbers = [13, 12, 11], # BL, BS, SO
            bfs_numbers = [
                2761, 2768,
                2546,  	 # Grenchen,  	
                           ],
            year_range = [2022, 2023],
            GWR_GKLAS = ['1110', ],  # '1121'],
            SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        ),

        DataAggScenario_Settings(
            name_dir_export = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI',
            # kt_numbers = [13, 12, 11], # BL, BS, SO
            kt_numbers = [13, 11],
            bfs_numbers = [
                2761, 2768,
                2546,  	 # Grenchen,  	
                           ],
            year_range = [2015, 2024],
            split_data_geometry_AND_slow_api = True,
            GWR_GKLAS = [ '1110',  '1121', '1122',  '1276', '1278' ],
            SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        ),



        # DataAggScenario_Settings(
        #     name_dir_export = 'preprep_BL_22to23_extSolkatEGID_aggrfarms',
        #     kt_numbers = [13,],
        #     year_range = [2022, 2023],
        #     GWR_GKLAS = [ '1110',  '1121', '1122',  '1276', '1278' ],
        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),
        # DataAggScenario_Settings(
        #     name_dir_export = 'preprep_BL_22to23_extSolkatEGID_singlehouse',
        #     kt_numbers = [13,],
        #     year_range = [2022, 2023],
        #     GWR_GKLAS = ['1110', ],  # '1121'],
        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),


        # DataAggScenario_Settings(
        #     name_dir_export = 'preprep_BLSO_22to23_extSolkatEGID_aggrfarms',
        #     kt_numbers = [13, 11],
        #     year_range = [2022, 2023],
        #     GWR_GKLAS = ['1110', '1121', '1122', '1276', '1278', ],
        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),
        # DataAggScenario_Settings(
        #     name_dir_export = 'preprep_BLSO_22to23_extSolkatEGID_singlehouse',
        #     kt_numbers = [13, 11],
        #     year_range = [2022, 2023],
        #     GWR_GKLAS = ['1110', ],  # '1121'],
        #     SOLKAT_cols_adjust_for_missEGIDs_to_solkat = ['FLAECHE', 'STROMERTRAG'],
        # ),

        ]

    for dataagg_scen in dataagg_scen_list:
        dataagg_class = DataAggScenario(dataagg_scen)

        dataagg_class.run_data_agg()


print('done')

