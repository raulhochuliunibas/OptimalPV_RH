import sys
import os as os
import pandas as pd
import geopandas as gpd
import  sqlite3
import shutil
import copy

from shapely.geometry import Point



# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code.auxiliary_functions import checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr



# ------------------------------------------------------------------------------------------------------
# SPLIT DATA AND GEOMETRY
# ------------------------------------------------------------------------------------------------------
def split_data_geometry(scen,):
    """
    Split data and geometry for all geo data frames for faster importing later on
    """
    
    # SETUP --------------------------------------
    print_to_logfile('run function: split_data_and_geometry.py', scen.log_name)
    os.makedirs(f'{scen.data_path}/input_split_data_geometry', exist_ok=True)


    # IMPORT DATA --------------------------------------   
    gm_shp_df = gpd.read_file(f'{scen.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

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
    elec_prod_gdf = gpd.read_file(f'{scen.data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', layer ='ElectricityProductionPlant')
    pv_all_gdf = copy.deepcopy(elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'])
    checkpoint_to_logfile(f'import pv, {pv_all_gdf.shape[0]} rows', scen.log_name, 1, scen.show_debug_prints)

    pv_all_gdf = attach_bfs_to_spatial_data(pv_all_gdf, gm_shp_df)
    pv_all_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)

    # split + export
    checkpoint_to_logfile(f'-- check unique identifier pv: {pv_all_gdf["xtf_id"].nunique()} xtf unique, {pv_all_gdf.shape[0]} rows', scen.log_name, 0, scen.show_debug_prints)
    pv_pq = copy.deepcopy(pv_all_gdf.loc[:,pv_all_gdf.columns !='geometry'])
    pv_geo = copy.deepcopy(pv_all_gdf.loc[:,['xtf_id', 'BFS_NUMMER', 'geometry']])

    pv_pq.to_parquet(f'{scen.data_path}/input_split_data_geometry/pv_pq.parquet')
    checkpoint_to_logfile('-- exported pv_pq.parquet', scen.log_name, 5, scen.show_debug_prints)

    with open(f'{scen.data_path}/input_split_data_geometry/pv_geo.geojson', 'w') as f:
        f.write(pv_geo.to_json())
    checkpoint_to_logfile('-- exported pv_geo.geojson', scen.log_name, 5, scen.show_debug_prints)


    # SOLKAT -------------------
    solkat_all_gdf = gpd.read_file(f'{scen.data_path}/input\solarenergie-eignung-daecher_2056.gpkg\SOLKAT_DACH.gpkg', layer ='SOLKAT_CH_DACH')
    checkpoint_to_logfile(f'import solkat, {solkat_all_gdf.shape[0]} rows', scen.log_name, 2, scen.show_debug_prints)

    solkat_all_gdf = attach_bfs_to_spatial_data(solkat_all_gdf, gm_shp_df)
    solkat_all_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)

    # split + export
    checkpoint_to_logfile(f'-- check unique identifier solkat: {solkat_all_gdf["DF_UID"].nunique()} DF_UID unique, {solkat_all_gdf.shape[0]} rows', scen.log_name, 5, scen.show_debug_prints)
    solkat_pq = copy.deepcopy(solkat_all_gdf.loc[:,solkat_all_gdf.columns !='geometry'])
    solkat_geo = copy.deepcopy(solkat_all_gdf.loc[:,['DF_UID', 'BFS_NUMMER', 'geometry']])

    solkat_pq.to_parquet(f'{scen.data_path}/input_split_data_geometry/solkat_pq.parquet')
    checkpoint_to_logfile('-- exported solkat_pq.parquet', scen.log_name, 5, scen.show_debug_prints)

    with open(f'{scen.data_path}/input_split_data_geometry/solkat_geo.geojson', 'w') as f:
        f.write(solkat_geo.to_json())
    checkpoint_to_logfile('-- exported solkat_geo.geojson', scen.log_name, 5, scen.show_debug_prints)


    # SOLKAT MONTH -------------------
    solkat_month_pq = gpd.read_file(f'{scen.data_path}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='SOLKAT_CH_DACH_MONAT')
    solkat_month_pq.to_parquet(f'{scen.data_path}/input_split_data_geometry/solkat_month_pq.parquet')
    


    # SUBSET for BSBLSO case ========================================================
    bsblso_bfs_numbers = get_bfs_from_ktnr([11, 12, 13,], scen.data_path, scen.log_name)

    # PV -------------------
    checkpoint_to_logfile('subset pv for bsblso case', scen.log_name, 5, scen.show_debug_prints)
    pv_bsblso_geo = copy.deepcopy(pv_geo.loc[pv_geo['BFS_NUMMER'].isin(bsblso_bfs_numbers)])
    if pv_bsblso_geo.shape[0] > 0:
        with open (f'{scen.data_path}/input_split_data_geometry/pv_bsblso_geo.geojson', 'w') as f:
            f.write(pv_bsblso_geo.to_json())
        checkpoint_to_logfile('-- exported pv_bsblso_geo.geojson', scen.log_name, 5, scen.show_debug_prints)

    # SOLKAT -------------------
    checkpoint_to_logfile('subset solkat for bsblso case', scen.log_name, 5, scen.show_debug_prints)
    solkat_bsblso_geo = copy.deepcopy(solkat_geo.loc[solkat_geo['BFS_NUMMER'].isin(bsblso_bfs_numbers)])
    if solkat_bsblso_geo.shape[0] > 0:
        with open (f'{scen.data_path}/input_split_data_geometry/solkat_bsblso_geo.geojson', 'w') as f:
            f.write(solkat_bsblso_geo.to_json())
        checkpoint_to_logfile('-- exported solkat_bsblso_geo.geojson', scen.log_name, 5, scen.show_debug_prints)

    # GWR -------------------
    # get all BUILDING data 
    # select cols
    query_columns = scen.GWR_building_cols
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in bsblso_bfs_numbers])

    conn = sqlite3.connect(f'{scen.data_path}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
    sqlrows = cur.fetchall()
    conn.close()
    checkpoint_to_logfile('sql query ALL BUILDING done', scen.log_name, 5, scen.show_debug_prints)

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
    gwr_bsblso_pq.to_parquet(f'{scen.data_path}/input_split_data_geometry/gwr_bsblso_pq.parquet')
    gwr_bsblso_gdf.to_file(f'{scen.data_path}/input_split_data_geometry/gwr_bsblso_gdf.geojson', driver='GeoJSON')


    # Copy Log File to input_split_data_geometry folder
    if os.path.exists(scen.log_name):
        shutil.copy(scen.log_name, f'{scen.data_path}/input_split_data_geometry/split_data_geometry_logfile.txt')




