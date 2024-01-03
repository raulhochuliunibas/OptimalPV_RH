import os as os
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import pyogrio
import winsound



# from functions import chapter_to_logfile, checkpoint_to_logfile
from datetime import datetime
from shapely.ops import unary_union
from shapely import wkb
from shapely.geometry import MultiPolygon


# ------------------------------------------------------------------------------------------------------
# LOG FIlE PRINTING FUNCTIONS
# ------------------------------------------------------------------------------------------------------

def chapter_to_logfile(str, log_file_name):
    """
    Function to write a chapter to the logfile
    """
    check = f'\n\n****************************************\n {str} \n start at:{datetime.now()} \n****************************************\n\n'
    print(check)
    with open(f'{log_file_name}', 'a') as log_file:
        log_file.write(f'{check}\n')
time_last_call = None
def checkpoint_to_logfile(str, log_file_name, n_tabs = 0, timer_func=None):
    """
    Function to write a checkpoint to the logfile
    """
    global time_last_call
    
    time_now = datetime.now()
    if time_last_call:
        runtime = time_now - time_last_call
        minutes, seconds = divmod(runtime.seconds, 60)
        runtime_str = f"{minutes} min {seconds} sec"
    else:
        runtime_str = 'N/A'
    
    n_tabs_str = '\t' * n_tabs
    check = f'* {str}{n_tabs_str}runtime: {runtime_str};   (stamp: {datetime.now()})'
    print(check)

    with open(f'{log_file_name}', 'a') as log_file:
        log_file.write(f"{check}\n")
    
    time_last_call = time_now



# ------------------------------------------------------------------------------------------------------
# DATA IMPORT + AGGREGATION
# ------------------------------------------------------------------------------------------------------

"""
Import data from input folder, subset for a relevant area and aggregate to building specific topology
Input:
- name: name of the aggregated data set
- script_run_on_server: 0 = script is running on laptop, 1 = script is running on server
- data_source: 'input' = import data from input folder, 'parquet' = import data from parquet folder
- gm_numbers_aggdef: list of municipality numbers to be considered
- select_gwr_aggdef: list of GWR types (number codes) to be considered from the building registry
- select_solkat_aggdef: list of Sonnendach types (number codes) to be considered from the building registry
Output: 
- aggregated data set including all the relevant data sources in a single output directory
"""
    
def import_aggregate_data(
    name_aggdef = "test_agg",
    script_run_on_server = 0,
    data_source = 'input', 
    gm_number_aggdef = None,
    select_gwr_aggdef = None,
    select_solkat_aggdef = None, 
    set_buffer = 1.25,
    ):

    # print('not funning function properly')
    # name_aggdef = "test_agg"
    # script_run_on_server = 0
    # data_source = 'parquet'
    # gm_number_aggdef = None
    # select_solkat_aggdef = None
    # select_gwr_aggdef = None
    # if True: 



    # ----------------------------------------------------------------------------------------------------------------
    # Setup + Import 
    # ----------------------------------------------------------------------------------------------------------------
       

    # pre setup + working + export directory -------------------------------------------------------------------------

    # set working directory
    if script_run_on_server == 0:
        winsound.Beep(840,  100)
        winsound.Beep(840,  100)
        wd_path = "C:\Models\OptimalPV_RH\data_aggregation"   # path for private computer
        data_path = "C:\Models\OptimalPV_RH_data"
    elif script_run_on_server == 1:
        wd_path = "D:\RaulHochuli_inuse\OptimalPV_RH\data_aggregation"         # path for server directory
        data_path = "D:/RaulHochuli_inuse\OptimalPV_RH_data"                  # path for server directory
    os.chdir(wd_path)

    # create new directory for export
    if not os.path.exists(f'{data_path}/{name_aggdef}'):
        os.makedirs(f'{data_path}/{name_aggdef}')
        
    # create log file for checkpoint comments
    timer = datetime.now()
    log_file_name_concat = f'{data_path}/{name_aggdef}/{name_aggdef}_log.txt'
    with open(f'{data_path}/{name_aggdef}/{name_aggdef}_log.txt', 'w') as log_file:
            log_file.write(f' \n')
    chapter_to_logfile(f'running local_data_import_aggregation.py - create: {name_aggdef}', log_file_name = log_file_name_concat)

    # --------------------------------------------------------------------------------------------------------------------
    # use PARQUET data for import directory ----
    # --------------------------------------------------------------------------------------------------------------------

    if data_source == 'parquet':
        if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
            print(f'ERROR: data_source = parquet, but no parquet data exists in {data_path}/spatial_intersection_by_gm')
        elif os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
            checkpoint_to_logfile(f'loading data FROM PARQUET', log_file_name = log_file_name_concat, n_tabs = 3)

        # import all spatial data set from parquet ---------------------------------------------------------------------
        if gm_number_aggdef is None:
            def import_parquet2gpd_all(file_name):
                """
                Function to import a parquet file to a geopandas dataframe
                Input: 
                - file_name: name of the parquet file to be imported
                Output:
                - geopandas dataframe
                """
                pd_df = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/{file_name}.parquet')
                pd_df['geometry'] = pd_df['geometry'].apply(wkb.loads)
                gpd_df = gpd.GeoDataFrame(pd_df, geometry='geometry')
                checkpoint_to_logfile(f'finished loading {file_name} parquet', log_file_name = log_file_name_concat, n_tabs = 1)
                return gpd_df
            
            gm_shp = import_parquet2gpd_all('gm_shp')
            roof_kat = import_parquet2gpd_all('roof_kat_by_gm')
            bldng_reg = import_parquet2gpd_all('bldng_reg_by_gm')
            heatcool_dem = import_parquet2gpd_all('heatcool_dem_by_gm')
            pv = import_parquet2gpd_all('pv_by_gm')


        # import a gm specific subset from parquet ------------------------------------------------------------------
        elif gm_number_aggdef is not None:
            is_gm_list_TF = isinstance(gm_number_aggdef, list)
            is_gm_intORfloat_TF = all(isinstance(elem, (int, float)) or np.issubdtype(type(elem), np.number) for elem in gm_number_aggdef)

            if  is_gm_list_TF and is_gm_intORfloat_TF :
                checkpoint_to_logfile(f'loading data FROM PARQUET by GM', log_file_name = log_file_name_concat, n_tabs = 2)

                def import_parquet2gpd_BY_GM(file_name):
                    """
                    Function to import a parquet file to a geopandas dataframe
                    Input: 
                    - file_name: name of the parquet file to be imported
                    Output:
                    - geopandas dataframe
                    """

                    filters = [('BFS_NUMMER', 'in', gm_number_aggdef)]
                    gm_shp_pq = pq.ParquetDataset(f'{data_path}/spatial_intersection_by_gm/{file_name}.parquet', filters=filters)
                    gm_shp_pd = gm_shp_pq.read_pandas().to_pandas()
                    gm_shp_pd['geometry'] = gm_shp_pd['geometry'].apply(wkb.loads)
                    gpd_df = gpd.GeoDataFrame(gm_shp_pd, geometry='geometry')
                    checkpoint_to_logfile(f'finished loading {file_name} parquet', log_file_name = log_file_name_concat, n_tabs = 1)
                    return gpd_df
                
                gm_shp = import_parquet2gpd_BY_GM('gm_shp')
                roof_kat = import_parquet2gpd_BY_GM('roof_kat_by_gm')
                bldng_reg = import_parquet2gpd_BY_GM('bldng_reg_by_gm')
                heatcool_dem = import_parquet2gpd_BY_GM('heatcool_dem_by_gm')
                pv = import_parquet2gpd_BY_GM('pv_by_gm')
 
        
        # ----------------------------------------------------------------------------------------------------------------
        # Transform + Subset by Relevant Municipalities 
        # ----------------------------------------------------------------------------------------------------------------

        # transform ------------------------------------------------------------------------------------------------------
        gm_shp_for_crs = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET', rows = 100)
        main_crs = gm_shp_for_crs.crs
        gm_shp.set_crs(main_crs, allow_override=True, inplace=True)
        roof_kat.set_crs(main_crs, allow_override=True, inplace=True)
        bldng_reg.set_crs(main_crs, allow_override=True, inplace=True)
        heatcool_dem.set_crs(main_crs, allow_override=True, inplace=True)
        pv.set_crs(main_crs, allow_override=True, inplace=True)
        all_crs_equal = gm_shp.crs == roof_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs

        if all_crs_equal:
            checkpoint_to_logfile(f'CRS are compatible', log_file_name = log_file_name_concat, n_tabs = 4)
        elif not all_crs_equal:
            checkpoint_to_logfile(f'CRS are NOT compatible', log_file_name = log_file_name_concat, n_tabs = 1)
            raise Exception('\nCRS are NOT compatible')
        
        # remove unnecessary columns for memory ------------------------------------------------------------------------------
        drop_cols_roof_kat = ['WAERMEERTRAG', 'DUSCHGAENGE', 'DG_HEIZUNG', 'DG_WAERMEBEDARF', 'BEDARF_WARMWASSER',
                              'BEDARF_HEIZUNG', 'FLAECHE_KOLLEKTOREN', 'VOLUMEN_SPEICHER', 'STROMERTRAG_SOMMERHALBJAHR', 
                              'STROMERTRAG_SOMMERHALBJAHR' ]
        drop_cols_bldng_reg = ['buildingStatus', 'buildingCategory', 'municipalityNumber', 'municipalityName', 'canton']
        drop_cols_heatcool_dem = ['NEEDSERVICE', 'NEEDTOTAL', 'NOGA', 'SERVICE', 'PERCENTGAS', 'PERCENTOIL',
                                  'PERCENTPUMP', 'PERCENTREMOTEHEAT', 'STYLE']
        drop_cols_pv = ['MainCategory', 'PlantCategory', ]
        drop_cols_gm_shp = ['UUID', 'DATUM_AEND', 'DATUM_ERST', 'ERSTELL_J', 'ERSTELL_M', 'REVISION_J', 'REVISION_M',
                            'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M', 'OBJEKTART', 'BEZIRKSNUM', 'SEE_FLAECH', 'REVISION_Q', 
                            'NAME',  'ICC', 'HIST_NR', 'GEM_TEIL', 'GEM_FLAECH', 'SHN', 'KANTONSNUM', 'EINWOHNERZ']
        
        roof_kat.drop(columns = drop_cols_roof_kat + ['index_right'] + drop_cols_gm_shp, axis = 1, inplace = True)
        bldng_reg.drop(columns = drop_cols_bldng_reg + ['index_right'] + drop_cols_gm_shp, axis = 1, inplace = True)
        heatcool_dem.drop(columns = drop_cols_heatcool_dem + ['index_right'] + drop_cols_gm_shp, axis = 1, inplace = True)
        pv.drop(columns = drop_cols_pv + ['index_right'] + drop_cols_gm_shp, axis = 1, inplace = True)
        gm_shp.drop(columns = drop_cols_gm_shp, axis = 1, inplace = True)   
        checkpoint_to_logfile(f'removed all unnecessary columns', log_file_name = log_file_name_concat, n_tabs = 3)

        # remove BFS_NUMMER column from all because it causes problems with export
        drop_BFS_NUMMER = ['BFS_NUMMER',]
        roof_kat.drop(columns = drop_BFS_NUMMER, axis = 1, inplace = True)
        bldng_reg.drop(columns = drop_BFS_NUMMER, axis = 1, inplace = True)
        heatcool_dem.drop(columns = drop_BFS_NUMMER, axis = 1, inplace = True)
        pv.drop(columns = drop_BFS_NUMMER, axis = 1, inplace = True)
        checkpoint_to_logfile(f'remove BFS columns from all but gm_shp', log_file_name = log_file_name_concat, n_tabs = 3)

        
        # subset roof_kat and bldng_reg for selected classes -------------------------------------------------------------

        # check if select_solkat_aggdef is a numeric list
        if select_solkat_aggdef is not None:
            is_solkat_list_TF = isinstance(select_solkat_aggdef, list)
            is_solkat_intORfloat_TF = all(isinstance(elem, (int, float)) or np.issubdtype(type(elem), np.number) for elem in select_solkat_aggdef)
            if is_solkat_list_TF and is_solkat_intORfloat_TF:
                roof_kat = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(select_solkat_aggdef)].copy()
                roof_kat['SB_OBJEKTART'].value_counts()
                # faca_kat = faca_kat.loc[faca_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
        
        # check if select_gwr_aggdef is a numeric list
        if select_gwr_aggdef is not None:
            is_gwr_list_TF = isinstance(select_gwr_aggdef, list)
            is_gwr_intORfloat_TF = all(isinstance(elem, (int, float)) or np.issubdtype(type(elem), np.number) for elem in select_gwr_aggdef)
            if is_gwr_list_TF and is_gwr_intORfloat_TF:
                bldng_reg = bldng_reg.loc[bldng_reg['buildingClass'].isin(select_gwr_aggdef)].copy()    

        # ----------------------------------------------------------------------------------------------------------------
        # Aggregate through Intersection 
        # ----------------------------------------------------------------------------------------------------------------

        # create house union shapes --------------------------------------------------------------------------------------
        # check if set_buffer is an integer
        if set_buffer > 0:
            # unionize buffered polygons
            checkpoint_to_logfile(f'start unionize buffered polygons', log_file_name = log_file_name_concat, n_tabs = 3)
            # set_buffer = 1.25
            roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
            roof_agg = gpd.GeoDataFrame(roof_agg_Srs, geometry=roof_agg_Srs)
            roof_agg.set_crs(main_crs, allow_override=True, inplace=True)
            checkpoint_to_logfile(f'finished unionize buffered polygons', log_file_name = log_file_name_concat, n_tabs = 3)

        elif set_buffer == False:
            
            def flatten_multipolygons(geoms):
                polygons = []
                for geom in geoms:
                    if isinstance(geom, MultiPolygon):
                        polygons.extend(list(geom))
                    else:
                        polygons.append(geom)
                return polygons
             
            # roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: MultiPolygon(flatten_multipolygons([poly.buffer(set_buffer, resolution = 16) for poly in x])))        
            # roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: MultiPolygon([poly.buffer(set_buffer, resolution = 16) for poly in x]))
            # roof_agg = gpd.GeoDataFrame(roof_agg_Srs, geometry=roof_agg_Srs)
            # roof_agg.set_crs(main_crs, allow_override=True, inplace=True)

            roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(unary_union)
            roof_agg = gpd.GeoDataFrame(roof_agg_Srs, geometry='geometry')
            roof_agg.set_crs(roof_kat.crs, allow_override=True, inplace=True)


        # intersection of data sets --------------------------------------------------------------------------------------
        # roof_kat.rename(columns={'index_right': 'index_roofkat'}, inplace=True)
        df_join1 = gpd.sjoin(roof_agg, roof_kat, how = "left", predicate = "intersects")
        df_join1.rename(columns={'index_right': 'index_roofkat'}, inplace=True)
        checkpoint_to_logfile(f'joined df1: roof_kat', log_file_name = log_file_name_concat, n_tabs = 4 )
        df_join2 = gpd.sjoin(df_join1, pv, how = "left", predicate = "intersects")
        df_join2.rename(columns={'index_right': 'index_pv'}, inplace=True)
        checkpoint_to_logfile(f'joined df2: pv', log_file_name = log_file_name_concat, n_tabs = 3)
        df_join3 = gpd.sjoin(df_join2, gm_shp, how = "left", predicate = "intersects")
        df_join3.rename(columns={'index_right': 'index_gm'}, inplace=True)
        # df_join3.drop(columns = ['SB_UUID',], axis = 1, inplace = True)
        checkpoint_to_logfile(f'joined df3: gm_shp', log_file_name = log_file_name_concat, n_tabs = 3)

        df_join4 = gpd.sjoin(df_join3, bldng_reg, how = "left", predicate = "intersects")
        df_join4.rename(columns={'index_right': 'index_bldng_reg'}, inplace=True)
        checkpoint_to_logfile(f'joined df4: bldng_reg', log_file_name = log_file_name_concat, n_tabs = 3)
        
        df_join4['centroids'] = df_join4.geometry.centroid
        df_join5 = gpd.sjoin(gpd.GeoDataFrame(df_join4, geometry='centroids'), heatcool_dem, how="left", predicate="intersects")
        df_join5.rename(columns={'index_right': 'index_heatcool_dem'}, inplace=True)
        df_join5.geometry = df_join4.geometry
        checkpoint_to_logfile(f'joined df5: heatcool_dem', log_file_name = log_file_name_concat, n_tabs = 3)
        

        # ----------------------------------------------------------------------------------------------------------------
        # Export 
        # ----------------------------------------------------------------------------------------------------------------
        
        roof_agg.to_parquet(f'{data_path}/{name_aggdef}/roof_agg_{name_aggdef}.parquet')
        df_join3.to_parquet(f'{data_path}/{name_aggdef}/df3_{name_aggdef}.parquet')
        df_join5.to_parquet(f'{data_path}/{name_aggdef}/df5_{name_aggdef}.parquet')  
        gm_shp.to_parquet(f'{data_path}/{name_aggdef}/{name_aggdef}_selected_gm_shp.parquet')
        checkpoint_to_logfile(f'exported df_join3 to parquet', log_file_name = log_file_name_concat, n_tabs = 3)    


        chapter_to_logfile(f'finished local_data_import_aggregation.py - create: {name_aggdef}', log_file_name = log_file_name_concat)




        


            
"""
    # --------------------------------------------------------------------------------------------------------------------
    # use local data from input directory ----
    # --------------------------------------------------------------------------------------------------------------------
    elif data_source == 'input':

        # import geo referenced data -------------------------------------------------------------------------------------
        subset_TF = True
        nrows = 1000
        if not subset_TF:
            # load administrative shapes
            kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
            gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
            checkpoint_to_logfile(f'finished loading administrative shp', log_file_name = log_file_name_concat, n_tabs = 2)

            # load solar kataster shapes
            roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
            checkpoint_to_logfile(f'finished loading roof solar kataster shp', log_file_name = log_file_name_concat, n_tabs = 1)
            #faca_kat = roof_kat.copy()
            #checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)

            # load building register indicating residential or industrial use
            bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
            checkpoint_to_logfile(f'finished loading building register', log_file_name = log_file_name_concat, n_tabs = 2)

            # load heating / cooling demand raster 150x150m
            heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
            checkpoint_to_logfile(f'finished loading heat & cool demand', log_file_name = log_file_name_concat, n_tabs = 1)

            # load pv installation points
            elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
            pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
            checkpoint_to_logfile(f'finished loading pv installation', log_file_name = log_file_name_concat, n_tabs = 2) 

        elif subset_TF:
            # load administrative shapes
            kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
            gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
            checkpoint_to_logfile(f'finished loading administrative shp', log_file_name = log_file_name_concat, n_tabs = 2)

            # load solar kataster shapes
            roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = nrows)
            roof_kat = roof_kat.apply(lambda col: col.astype(str) if col.dtype == 'datetime64[ns]' else col)
            roof_kat.to_file(f'{data_path}/temp_cache/solkat_{nrows}.geojson', driver='GeoJSON')  # GeoJSON format

            checkpoint_to_logfile(f'finished loading roof solar kataster shp', log_file_name = log_file_name_concat, n_tabs = 1)
            #faca_kat = roof_kat.copy()
            #checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)

            # load building register indicating residential or industrial use
            bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson', rows = 1000)
            checkpoint_to_logfile(f'finished loading building register', log_file_name = log_file_name_concat, n_tabs = 2)

            # load heating / cooling demand raster 150x150m
            heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES', rows = 1000)
            checkpoint_to_logfile(f'finished loading heat & cool demand', log_file_name = log_file_name_concat, n_tabs = 1)

            # load pv installation points
            elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 1000)
            pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
            checkpoint_to_logfile(f'finished loading pv installation', log_file_name = log_file_name_concat, n_tabs = 2)


        # ----------------------------------------------------------------------------------------------------------------
        # Transform + Subset by Relevant Municipalities 
        # ----------------------------------------------------------------------------------------------------------------

        # transform ------------------------------------------------------------------------------------------------------

        # check if all CRS are compatible
        main_crs = gm_shp.crs
        kt_shp.crs == gm_shp.crs == roof_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs
        gm_shp.set_crs(main_crs, allow_override=True, inplace=True)
        roof_kat.set_crs(main_crs, allow_override=True, inplace=True)
        # faca_kat.set_crs(main_crs, allow_override=True, inplace=True)
        bldng_reg.set_crs(main_crs, allow_override=True, inplace=True)
        heatcool_dem.set_crs(main_crs, allow_override=True, inplace=True)
        pv.set_crs(main_crs, allow_override=True, inplace=True)
            
        all_crs_equal = kt_shp.crs == gm_shp.crs == roof_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs
        if all_crs_equal:
            checkpoint_to_logfile(' - ', log_file_name = log_file_name_concat, n_tabs = 6)
            checkpoint_to_logfile(f'CRS are compatible', log_file_name = log_file_name_concat, n_tabs = 4)
        elif not all_crs_equal:
            checkpoint_to_logfile(f'CRS are NOT compatible', log_file_name = log_file_name_concat, n_tabs = 1)
            raise Exception('\nCRS are NOT compatible')
        

        # transform to minimize memory & remove columns not needed and  --------------------------------------------------
        def minimize_type_for_memory(df):
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
                elif pd.api.types.is_object_dtype(df[col]):
                    df[col] = df[col].astype(str)
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = df[col].astype('float32')
            return df
        

        roof_kat = minimize_type_for_memory(roof_kat)
        drop_cols_roof_kat = ['WAERMEERTRAG', 'DUSCHGAENGE', 'DG_HEIZUNG', 'DG_WAERMEBEDARF', 'BEDARF_WARMWASSER',
                    'BEDARF_HEIZUNG', 'FLAECHE_KOLLEKTOREN', 'VOLUMEN_SPEICHER', 'STROMERTRAG_SOMMERHALBJAHR', 'STROMERTRAG_SOMMERHALBJAHR' ]
        roof_kat.drop(columns=drop_cols_roof_kat, axis=1, inplace=True)
        roof_kat.info()

        bldng_reg = minimize_type_for_memory(bldng_reg)
        drop_cols_bldng_reg = ['buildingStatus', 'buildingCategory', 'municipalityNumber', 'municipalityName',
                    'canton']
        bldng_reg.drop(columns=drop_cols_bldng_reg, axis=1, inplace=True)    
        bldng_reg.info()

        heatcool_dem = minimize_type_for_memory(heatcool_dem)
        drop_cols_heatcool_dem = ['NEEDSERVICE', 'NEEDTOTAL', 'NOGA', 'SERVICE', 'PERCENTGAS', 'PERCENTOIL', 
                    'PERCENTPUMP', 'PERCENTREMOTEHEAT', 'STYLE']
        heatcool_dem.drop(columns=drop_cols_heatcool_dem, axis=1, inplace=True)
        heatcool_dem.info()
        
        pv = minimize_type_for_memory(pv)
        drop_cols_pv = ['MainCategory', 'PlantCategory', ]
        pv.drop(columns=drop_cols_pv, axis=1, inplace=True)

        gm_shp = minimize_type_for_memory(gm_shp)
        drop_cols_gm_shp = ['UUID', 'DATUM_AEND', 'DATUM_ERST', 'ERSTELL_J', 'ERSTELL_M', 'REVISION_J', 
                    'REVISION_M', 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M', 'OBJEKTART', 
                    'REVISION_Q', 'ICC', 'GEM_TEIL', 'GEM_FLAECH', 'SHN', 'SEE_FLAECH']
        gm_shp.drop(columns=drop_cols_gm_shp, axis=1, inplace=True)
        gm_shp.info()
        checkpoint_to_logfile(f'dropped unnecessary columns', log_file_name = log_file_name_concat, n_tabs = 3)


        # subset roof_kat and bldng_reg for selected classes -------------------------------------------------------------

        # check if select_solkat_aggdef is a numeric list
        if select_solkat_aggdef is not None:
            is_solkat_list_TF = isinstance(select_solkat_aggdef, list)
            is_solkat_intORfloat_TF = all(isinstance(elem, (int, float)) or np.issubdtype(type(elem), np.number) for elem in select_solkat_aggdef)
            if is_solkat_list_TF and is_solkat_intORfloat_TF:
                roof_kat = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(select_solkat_aggdef)].copy()
                roof_kat['SB_OBJEKTART'].value_counts()
                # faca_kat = faca_kat.loc[faca_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
        
        # check if select_gwr_aggdef is a numeric list

        if select_gwr_aggdef is not None:
            is_gwr_list_TF = isinstance(select_gwr_aggdef, list)
            is_gwr_intORfloat_TF = all(isinstance(elem, (int, float)) or np.issubdtype(type(elem), np.number) for elem in select_gwr_aggdef)
            if is_gwr_list_TF and is_gwr_intORfloat_TF:
                bldng_reg = bldng_reg.loc[bldng_reg['buildingClass'].isin(select_gwr_aggdef)].copy()    


        # subset by selected gm shp --------------------------------------------------------------------------------------
        
        # check if gm_number_aggdef is a numeric list
        if gm_number_aggdef is not None:
            is_gm_list_TF = isinstance(gm_number_aggdef, list)
            is_gm_intORfloat_TF = all(isinstance(elem, (int, float)) or np.issubdtype(type(elem), np.number) for elem in gm_number_aggdef)
            if  is_gm_list_TF and is_gm_intORfloat_TF :
                subset_shape = gm_shp.loc[gm_shp['BFS_NUMMER'].isin(gm_number_aggdef),].copy()
                
                roof_kat = gpd.sjoin(roof_kat, subset_shape, how="inner", predicate="within")
                checkpoint_to_logfile(f'subset roof_kat for gm selection', log_file_name = log_file_name_concat, n_tabs = 2)
                bldng_reg = gpd.sjoin(bldng_reg, subset_shape, how="inner", predicate="within")
                checkpoint_to_logfile(f'subset bldng_reg for gm selection', log_file_name = log_file_name_concat, n_tabs = 2)
                heatcool_dem = gpd.sjoin(heatcool_dem, subset_shape, how="inner", predicate="within")
                checkpoint_to_logfile(f'subset heatcool_dem for gm selection', log_file_name = log_file_name_concat, n_tabs = 2)
                pv = gpd.sjoin(pv, subset_shape, how="inner", predicate="within")
                checkpoint_to_logfile(f'subset pv for gm selection', log_file_name = log_file_name_concat, n_tabs = 3)

                roof_kat.drop(columns="index_right", inplace=True)
                bldng_reg.drop(columns="index_right", inplace=True)
                heatcool_dem.drop(columns="index_right", inplace=True)
                pv.drop(columns="index_right", inplace=True)
                print('\n')



        # ----------------------------------------------------------------------------------------------------------------
        # Aggregate through Intersection 
        # ----------------------------------------------------------------------------------------------------------------

        # create house union shapes --------------------------------------------------------------------------------------
        # unionize buffered polygons
        set_buffer = 1.25
        roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
        roof_agg = gpd.GeoDataFrame(roof_agg_Srs, geometry=roof_agg_Srs)
        roof_agg.set_crs(main_crs, allow_override=True, inplace=True)


        # intersection of data sets --------------------------------------------------------------------------------------
        # roof_kat.rename(columns={'index_right': 'index_roofkat'}, inplace=True)
        df_join1 = gpd.sjoin(roof_agg, roof_kat, how = "left", predicate = "intersects")
        df_join1.rename(columns={'index_right': 'index_roofkat'}, inplace=True)
        checkpoint_to_logfile(f'joined df1: roof_kat', log_file_name = log_file_name_concat, n_tabs = 4 )
        df_join2 = gpd.sjoin(df_join1, pv, how = "left", predicate = "intersects")
        df_join2.rename(columns={'index_right': 'index_pv'}, inplace=True)
        checkpoint_to_logfile(f'joined df2: pv', log_file_name = log_file_name_concat, n_tabs = 3)
        df_join3 = gpd.sjoin(df_join2, gm_shp, how = "left", predicate = "intersects")
        df_join3.rename(columns={'index_right': 'index_gm'}, inplace=True)
        df_join3.drop(columns = ['SB_UUID',], axis = 1, inplace = True)
        checkpoint_to_logfile(f'joined df3: gm_shp', log_file_name = log_file_name_concat, n_tabs = 3)

        # df_join4 = gpd.sjoin(df_join3, bldng_reg, how = "left", predicate = "intersects")
        # df_join4.rename(columns={'index_right': 'index_bldng_reg'}, inplace=True)
        # checkpoint_to_logfile(f'joined df4: bldng_reg', log_file_name = log_file_name_concat, n_tabs = 3)
        # df_join5 = gpd.sjoin(df_join4, heatcool_dem, how = "left", predicate = "intersects")
        # df_join5.rename(columns={'index_right': 'index_heatcool_dem'}, inplace=True)
        # checkpoint_to_logfile(f'joined df5: heatcool_dem', log_file_name = log_file_name_concat, n_tabs = 3)


        # ----------------------------------------------------------------------------------------------------------------
        # Export 
        # ----------------------------------------------------------------------------------------------------------------

        df_join3.to_file(f'{data_path}/{name_aggdef}/{name_aggdef}_solkat_pv_gm_gdf.geojson', driver='GeoJSON')  # GeoJSON format
        df_join3.to_parquet(f'{data_path}/{name_aggdef}/{name_aggdef}_solkat_pv_gm.parquet')  
        checkpoint_to_logfile(f'exported df_join3 to geojson', log_file_name = log_file_name_concat, n_tabs = 3)
        # df_join5.to_file(f'{data_path}/{name_aggdef}/{name_aggdef}_MAIN_gdf.geojson', driver='GeoJSON')  # GeoJSON format
        if isinstance(subset_shape, gpd.GeoDataFrame):
            subset_shape.to_file(f'{data_path}/{name_aggdef}/{name_aggdef}_subset_shape.geojson', driver='GeoJSON') 

        # heatcool_dem.to_file(f'{data_path}/{name_aggdef}/heatcool_dem.geojson', driver='GeoJSON')  # GeoJSON format

    

        if script_run_on_server == 0:
            winsound.Beep(840,  100)
            winsound.Beep(840,  100)

"""


