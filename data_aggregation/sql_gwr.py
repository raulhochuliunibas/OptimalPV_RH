import sys
import os
import pandas as pd
import geopandas as gpd
import  sqlite3
import re

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# SQL DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# GEBÄUDE- UND WOHNUNGSREGISTER --------------------------------------------
#> https://www.housing-stat.ch/de/madd/public.html
def sql_gwr_data(
        dataagg_settings_def, ):
    """
    Function to import data from the Building and Dwelling (Gebaeude und Wohungsregister) database.
    Import data from SQL file and save the relevant variables locally as parquet file.
    """ 

    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']

    gwr_selection_specs_def = dataagg_settings_def['gwr_selection_specs']
    print_to_logfile('run function: sql_gwr_data.py', log_file_name_def=log_file_name_def)


    # querys -------------------
    # Get all column names from GWR 2.0
    if False: 
        print_to_logfile('check all GWR cols for empty cells', log_file_name_def=log_file_name_def)
        col_nan_table, col_nan_name, col_nan_count = [], [], []

        conn = sqlite3.connect(f'{data_path_def}/input/GebWohnRegister.CH/data.sqlite')
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        for table in tables:
            table_name = table[0]
            
            # Get the column names and count ""
            cur.execute(f"PRAGMA table_info({table_name});")
            columns = cur.fetchall()
            checkpoint_to_logfile(f'check table: {table_name}, no. of cols: {len(columns)}', log_file_name_def = log_file_name_def, n_tabs_def = 5)


            column_names = [column[1] for column in columns]
            for column_name in column_names:
                cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} = '';")
                empty_count = cur.fetchone()[0]
                
                col_nan_table.append(table_name)
                col_nan_name.append(column_name)
                col_nan_count.append(empty_count)
        
        conn.close()
        gwr_nan_df = pd.DataFrame({'table': col_nan_table, 'col': col_nan_name, 'empty_cells': col_nan_count})
        gwr_nan_df.to_csv(f'{wd_path_def}/gwr_nan.csv', sep=';', index=False) 
        gwr_nan_df.to_csv(f'{data_path_def}/output/preprep_data/gwr_nan.csv', sep=';', index=False)
        checkpoint_to_logfile('exported gwr_nan data', log_file_name_def=log_file_name_def, n_tabs_def = 5)


    # get BUILDING data
    # select cols
    query_columns = gwr_selection_specs_def['building_cols']
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in bfs_number_def])
    # select rows
    gklas_values = gwr_selection_specs_def['GKLAS']
    if isinstance(gklas_values, list):
        gklas_values_str = ', '.join([f"'{str(value)}'" for value in gklas_values])
    else:
        gklas_values_str = f"'{str(gklas_values)}'"

    conn = sqlite3.connect(f'{data_path_def}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers}) AND GKLAS IN ({gklas_values_str})')
    sqlrows = cur.fetchall()
    conn.close()
    checkpoint_to_logfile('sql query BUILDING done', log_file_name_def, 10, show_debug_prints_def)

    gwr_building_df = pd.DataFrame(sqlrows, columns=query_columns)
    gwr_building_df.to_csv(f'{data_path_def}/output/preprep_data/gwr_building_df.csv', sep=';', index=False)
    selected_EGID = list(gwr_building_df['EGID'])


    # get DWELLING data
    # select cols
    query_columns = gwr_selection_specs_def['dwelling_cols']
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in bfs_number_def])
    # select rows
    egid_values = selected_EGID
    # egid_values_str = ', '.join([f"'{str(value)}'" for value in egid_values])

    conn = sqlite3.connect(f'{data_path_def}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM dwelling')
    sqlrows = cur.fetchall()
    conn.close()
    checkpoint_to_logfile('sql query DWELLING done', log_file_name_def, 10, show_debug_prints_def)

    gwr_dwelling_df = pd.DataFrame(sqlrows, columns=query_columns)


    # merger & filter for specs -------------------
    gwr = gwr_building_df.merge(gwr_dwelling_df, on='EGID', how='inner')

    checkpoint_to_logfile(f'check gwr BEFORE filtering: {gwr["EGID"].nunique()} unique EGIDs in gwr.shape {gwr.shape}, {round(gwr["EGID"].nunique()/gwr.shape[0],2)*100} %', log_file_name_def, 3, True)
    gwr['GBAUJ'] = gwr['GBAUJ'].replace('', 0).astype(int)
    gwr = gwr[(gwr['GSTAT'].isin(gwr_selection_specs_def['GSTAT'])) & 
              (gwr['GKLAS'].isin(gwr_selection_specs_def['GKLAS'])) & 
              (gwr['GBAUJ'] >= gwr_selection_specs_def['GBAUJ_minmax'][0]) &
              (gwr['GBAUJ'] <= gwr_selection_specs_def['GBAUJ_minmax'][1])]
    gwr['GBAUJ'] = gwr['GBAUJ'].replace(0, '').astype(str)
    checkpoint_to_logfile(f'check gwr AFTER filtering: {gwr["EGID"].nunique()} unique EGIDs in gwr.shape {gwr.shape}, {round(gwr["EGID"].nunique()/gwr.shape[0],2)*100} %', log_file_name_def, 3, True)


    # check proxy possiblity -------------------
    checkpoint_to_logfile(f'gwr_guilding_df.shape: {gwr_building_df.shape}, EGID: {gwr_building_df["EGID"].nunique()};\n  gwr_dwelling_df.shape: {gwr_dwelling_df.shape}, EGID: {gwr_dwelling_df["EGID"].nunique()};\n  gwr.shape: {gwr.shape}, EGID: {gwr["EGID"].nunique()}', log_file_name_def, 2, True)
    
    checkpoint_to_logfile(f'* check for WAZIM: {gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAZIM", {round(gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique() / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', log_file_name_def, 1, True)
    checkpoint_to_logfile(f'* check for WAREA: {gwr.loc[gwr["WAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAREA", {round(gwr.loc[gwr["WAREA"] != "", "EGID"].nunique() / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', log_file_name_def, 1, True)
    checkpoint_to_logfile(f'* check for GAREA: {gwr.loc[gwr["GAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty GAREA", {round(gwr.loc[gwr["GAREA"] != "", "EGID"].nunique() / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', log_file_name_def, 1, True)
    
    # checkpoint_to_logfile('Did NOT us a combination of building and dwelling data, \n because they overlap way too little. This makes sense \nintuitievly as single unit houses probably are not registered \nas dwellings in the data base.', log_file_name_def, 1, True)


    # merge dfs and export -------------------
    # gwr = gwr_building_df
    gwr.to_csv(f'{data_path_def}/output/preprep_data/gwr.csv', sep=';', index=False)
    gwr.to_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
    checkpoint_to_logfile(f'exported gwr data', log_file_name_def=log_file_name_def, n_tabs_def = 4)


    # create spatial df and export -------------------
    gwr = gwr.loc[(gwr['GKODE'] != '') & (gwr['GKODN'] != '')]
    gwr[['GKODE', 'GKODN']] = gwr[['GKODE', 'GKODN']].astype(float)
    gwr['geometry'] = gwr.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
    gwr_gdf = gpd.GeoDataFrame(gwr, geometry='geometry')
    gwr_gdf.crs = 'EPSG:2056'
    gwr_gdf = gwr_gdf.loc[:, ['EGID', 'geometry']]
    gwr_gdf.to_file(f'{data_path_def}/split_data_geometry/gwr_gdf.geojson', driver='GeoJSON')


