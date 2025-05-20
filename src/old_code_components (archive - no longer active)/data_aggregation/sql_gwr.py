import sys
import os
import pandas as pd
import geopandas as gpd
import  sqlite3
import copy

from shapely.geometry import Point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr


# ------------------------------------------------------------------------------------------------------
# SQL DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# GEBÄUDE- UND WOHNUNGSREGISTER --------------------------------------------
#> https://www.housing-stat.ch/de/madd/public.html
def sql_gwr_data(
        scen, ):
    """
    Function to import data from the Building and Dwelling (Gebaeude und Wohungsregister) database.
    Import data from SQL file and save the relevant variables locally as parquet file.
    """ 
    # SETUP --------------------------------------
    print_to_logfile('run function: sql_gwr_data.py', scen.log_name)


    # QUERYs --------------------------------------

    # get DWELLING data
    # select cols
    query_columns = scen.GWR_dwelling_cols
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in scen.bfs_numbers])

    conn = sqlite3.connect(f'{scen.data_path}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM dwelling')
    sqlrows = cur.fetchall()
    conn.close()
    checkpoint_to_logfile('sql query DWELLING done', scen.log_name, 10, scen.show_debug_prints)

    gwr_dwelling_df = pd.DataFrame(sqlrows, columns=query_columns)
    gwr_dwelling_df[['WAZIM', 'WAREA']] = gwr_dwelling_df[['WAZIM', 'WAREA']].replace('', 0).astype(float)
    gwr_dwelling_df.to_csv(f'{scen.preprep_path}/gwr_dwelling_df.csv', sep=';', index=False)


    # get ALL BUILDING data
    # select cols
    query_columns = scen.GWR_building_cols
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in scen.bfs_numbers])

    conn = sqlite3.connect(f'{scen.data_path}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
    sqlrows = cur.fetchall()
    conn.close()
    checkpoint_to_logfile('sql query ALL BUILDING done', scen.log_name, 10, scen.show_debug_prints)

    gwr_all_building_df = pd.DataFrame(sqlrows, columns=query_columns)
    gwr_all_building_df.to_csv(f'{scen.preprep_path}/gwr_all_building_df.csv', sep=';', index=False)
    gwr_all_building_df.to_parquet(f'{scen.preprep_path}/gwr_all_building_df.parquet')


    # merger -------------------
    # gwr = gwr_building_df.merge(gwr_dwelling_df, on='EGID', how='left')
    gwr_mrg = gwr_all_building_df.merge(gwr_dwelling_df, on='EGID', how='left')


    # aggregate dwelling data per EGID -------------------
    print('print to log_file')
    print_to_logfile('aggregate dwelling data per EGID', scen.log_name)
    checkpoint_to_logfile(f'check gwr BEFORE aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', scen.log_name, 3, True)

    print('print to summary_file')
    print_to_logfile('aggregate dwelling data per EGID', scen.summary_name)
    checkpoint_to_logfile(f'check gwr BEFORE aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', scen.summary_name, 3, True)

    bldg_agg_cols = copy.deepcopy(scen.GWR_building_cols)
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

    checkpoint_to_logfile(f'check gwr AFTER aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', scen.log_name, 3, True)
    checkpoint_to_logfile(f'check gwr AFTER aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],1)*100} %', scen.summary_name, 3, True)


    # filter for specs -------------------
    checkpoint_to_logfile(f'check gwr_mrg BEFORE filtering: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique() )/gwr_mrg.shape[0],2)*100} %', scen.log_name, 3, True)

    gwr_mrg0 = copy.deepcopy(gwr_mrg)
    gwr_mrg0['GBAUJ'] = gwr_mrg0['GBAUJ'].replace('', 0).astype(int)
    gwr_mrg1 = gwr_mrg0[(gwr_mrg0['GSTAT'].isin(scen.GWR_GSTAT))]
    gwr_mrg2 = gwr_mrg1[(gwr_mrg1['GKLAS'].isin(scen.GWR_GKLAS))]
    gwr_mrg3 = gwr_mrg2[(gwr_mrg2['GBAUJ'] >= scen.GWR_GBAUJ_minmax[0]) & (gwr_mrg2['GBAUJ'] <= scen.GWR_GBAUJ_minmax[1])]
    gwr = copy.deepcopy(gwr_mrg3)
    checkpoint_to_logfile(f'check gwr AFTER filtering: {gwr["EGID"].nunique()} unique EGIDs in gwr.shape {gwr.shape}, {round((gwr["EGID"].nunique() )/gwr_mrg.shape[0],2)*100} %', scen.log_name, 3, True)
    print_to_logfile('\n', scen.summary_name)


    # summary log -------------------
    print_to_logfile('Building and Dwelling data import:', scen.summary_name)
    checkpoint_to_logfile(f'gwr_mrg0.shape: {gwr_mrg0.shape}, EGID: {gwr_mrg0["EGID"].nunique()}', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'\t> selection range n BFS municipalities: {len(scen.bfs_numbers)}', scen.summary_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_name)
    checkpoint_to_logfile(f'after GSTAT selection, gwr.shape: {gwr_mrg1.shape} EGID.nunique: {gwr_mrg1["EGID"].nunique()} ({round((gwr_mrg1.shape[0] ) / gwr_mrg0.shape[0] * 100, 2)} % of gwr_mrg0)', scen.summary_name, 2, True) 
    checkpoint_to_logfile(f'\t> selection GSTAT: {scen.GWR_GSTAT} "only existing bulidings"', scen.summary_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_name)
    checkpoint_to_logfile(f'after GKLAS selection, gwr.shape: {gwr_mrg2.shape} EGID.nunique: {gwr_mrg2["EGID"].nunique()} ({round((gwr_mrg2.shape[0] ) / gwr_mrg1.shape[0] * 100, 2)} % of gwr_mrg1)', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'\t> selection GKLAS: {scen.GWR_GKLAS} "1110 - building w 1 living space, 1121 - w 2 living spaces, 1276 - agricluture buildings (stables, barns )"', scen.summary_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_name)
    checkpoint_to_logfile(f'after GBAUJ_minmax selection, gwr.shape: {gwr_mrg3.shape} EGID.nunique: {gwr_mrg3["EGID"].nunique()} ({round((gwr_mrg3.shape[0] ) / gwr_mrg2.shape[0] * 100, 2)} % of gwr_mrg3)', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'\t> selection GBAUJ_minmax: {scen.GWR_GBAUJ_minmax} "built construction between years"', scen.summary_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_name)
    checkpoint_to_logfile(f'from ALL gwr_mrg0 (aggregated with dwelling, bfs already selected): {gwr_mrg0["EGID"].nunique()-gwr_mrg3["EGID"].nunique()} of {gwr_mrg0["EGID"].nunique()} EGIDs removed ({round((gwr_mrg0["EGID"].nunique() - gwr_mrg3["EGID"].nunique()  )/gwr_mrg0["EGID"].nunique()*100, 2)}%, mrg0-mrg3 of mrg0)', scen.summary_name, 2, True)
    checkpoint_to_logfile(f'\t> {gwr_mrg3["EGID"].nunique()} gwr_mrg3 EGIDS {round((gwr_mrg3["EGID"].nunique())/gwr_mrg0["EGID"].nunique()*100, 2)}%  of  {gwr_mrg0["EGID"].nunique()} gwr_mrg0', scen.summary_name, 2, True)
    print_to_logfile('\n', scen.summary_name)


    # check proxy possiblity -------------------
    # checkpoint_to_logfile(f'gwr_guilding_df.shape: {gwr_building_df.shape}, EGID: {gwr_building_df["EGID"].nunique()};\n  gwr_dwelling_df.shape: {gwr_dwelling_df.shape}, EGID: {gwr_dwelling_df["EGID"].nunique()};\n  gwr.shape: {gwr.shape}, EGID: {gwr["EGID"].nunique()}', scen.log_name, 2, True)
    
    checkpoint_to_logfile(f'* check for WAZIM: {gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAZIM", {round((gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', scen.log_name, 1, True)
    checkpoint_to_logfile(f'* check for WAREA: {gwr.loc[gwr["WAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAREA", {round((gwr.loc[gwr["WAREA"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', scen.log_name, 1, True)
    checkpoint_to_logfile(f'* check for GAREA: {gwr.loc[gwr["GAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty GAREA", {round((gwr.loc[gwr["GAREA"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', scen.log_name, 1, True)

    # checkpoint_to_logfile('Did NOT us a combination of building and dwelling data, \n because they overlap way too little. This makes sense \nintuitievly as single unit houses probably are not registered \nas dwellings in the data base.', log_file_name_def, 1, True)


    # merge dfs and export -------------------
    # gwr = gwr_building_df
    gwr.to_csv(f'{scen.preprep_path}/gwr.csv', sep=';', index=False)
    gwr_mrg_all_building_in_bfs.to_csv(f'{scen.preprep_path}/gwr_mrg_all_building_in_bfs.csv', sep=';', index=False)
    gwr.to_parquet(f'{scen.preprep_path}/gwr.parquet')
    gwr_mrg_all_building_in_bfs.to_parquet(f'{scen.preprep_path}/gwr_mrg_all_building_in_bfs.parquet')
    checkpoint_to_logfile('exported gwr data', scen.log_name, n_tabs_def = 4)


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
    gwr_gdf.to_file(f'{scen.preprep_path}/gwr_gdf.geojson', driver='GeoJSON')

    # gwr_all_building_gdf exported for DSO nodes location determination later
    gwr_all_building_gdf = gwr_to_gdf(gwr_all_building_df)
    gwr_all_building_gdf.to_file(f'{scen.preprep_path}/gwr_all_building_gdf.geojson', driver='GeoJSON')

    if scen.split_data_geometry_AND_slow_api:
        gwr_gdf.to_file(f'{scen.data_path}/input_split_data_geometry/gwr_gdf.geojson', driver='GeoJSON')

