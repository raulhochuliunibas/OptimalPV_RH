import sys
import os
import pandas as pd
import geopandas as gpd
import  sqlite3
import copy

from shapely.geometry import Point

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
    summary_file_name = dataagg_settings_def['summary_file_name']

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
    gwr_dwelling_df[['WAZIM', 'WAREA']] = gwr_dwelling_df[['WAZIM', 'WAREA']].replace('', 0).astype(float)
    gwr_dwelling_df.to_csv(f'{data_path_def}/output/preprep_data/gwr_dwelling_df.csv', sep=';', index=False)


    # get ALL BUILDING data
    # select cols
    query_columns = gwr_selection_specs_def['building_cols']
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in bfs_number_def])

    conn = sqlite3.connect(f'{data_path_def}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
    sqlrows = cur.fetchall()
    conn.close()
    checkpoint_to_logfile('sql query ALL BUILDING done', log_file_name_def, 10, show_debug_prints_def)

    gwr_all_building_df = pd.DataFrame(sqlrows, columns=query_columns)
    gwr_all_building_df.to_csv(f'{data_path_def}/output/preprep_data/gwr_building_df.csv', sep=';', index=False)


    # merger -------------------
    # gwr = gwr_building_df.merge(gwr_dwelling_df, on='EGID', how='left')
    gwr_mrg = gwr_all_building_df.merge(gwr_dwelling_df, on='EGID', how='left')


    # aggregate dwelling data per EGID -------------------
    print_to_logfile('aggregate dwelling data per EGID', log_file_name_def)
    print_to_logfile('aggregate dwelling data per EGID', summary_file_name)
    checkpoint_to_logfile(f'check gwr BEFORE aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],2)*100} %', log_file_name_def, 3, True)
    checkpoint_to_logfile(f'check gwr BEFORE aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],2)*100} %', summary_file_name, 3, True)

    bldg_agg_cols = copy.deepcopy(gwr_selection_specs_def['building_cols'])
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

    checkpoint_to_logfile(f'check gwr AFTER aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],2)*100} %', log_file_name_def, 3, True)
    checkpoint_to_logfile(f'check gwr AFTER aggregation: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique())/gwr_mrg.shape[0],2)*100} %', summary_file_name, 3, True)


    # filter for specs -------------------
    checkpoint_to_logfile(f'check gwr_mrg BEFORE filtering: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique() )/gwr_mrg.shape[0],2)*100} %', log_file_name_def, 3, True)
    # gwr_mrg['GBAUJ'] = gwr_mrg['GBAUJ'].replace('', 0).astype(int)
    # gwr_mrg = gwr_mrg[(gwr_mrg['GSTAT'].isin(gwr_selection_specs_def['GSTAT'])) & 
    #           (gwr_mrg['GKLAS'].isin(gwr_selection_specs_def['GKLAS'])) & 
    #           (gwr_mrg['GBAUJ'] >= gwr_selection_specs_def['GBAUJ_minmax'][0]) &
    #           (gwr_mrg['GBAUJ'] <= gwr_selection_specs_def['GBAUJ_minmax'][1])]
    # gwr_mrg['GBAUJ'] = gwr_mrg['GBAUJ'].replace(0, '').astype(str)
    gwr_mrg0 = copy.deepcopy(gwr_mrg)
    gwr_mrg0['GBAUJ'] = gwr_mrg0['GBAUJ'].replace('', 0).astype(int)
    gwr_mrg1 = gwr_mrg0[(gwr_mrg0['GSTAT'].isin(gwr_selection_specs_def['GSTAT']))]
    gwr_mrg2 = gwr_mrg1[(gwr_mrg1['GKLAS'].isin(gwr_selection_specs_def['GKLAS']))]
    gwr_mrg3 = gwr_mrg2[(gwr_mrg2['GBAUJ'] >= gwr_selection_specs_def['GBAUJ_minmax'][0]) & (gwr_mrg2['GBAUJ'] <= gwr_selection_specs_def['GBAUJ_minmax'][1])]
    gwr_mrg = gwr_mrg3
    checkpoint_to_logfile(f'check gwr_mrg AFTER filtering: {gwr_mrg["EGID"].nunique()} unique EGIDs in gwr_mrg.shape {gwr_mrg.shape}, {round((gwr_mrg["EGID"].nunique() )/gwr_mrg.shape[0],2)*100} %', log_file_name_def, 3, True)
    print_to_logfile(f'\n', log_file_name_def=summary_file_name)


    # summary log -------------------
    print_to_logfile(f'Building and Dwelling data import:', summary_file_name)
    checkpoint_to_logfile(f'gwr_mrg0.shape: {gwr_mrg0.shape}, EGID: {gwr_mrg0["EGID"].nunique()}', summary_file_name, 2, True)
    # checkpoint_to_logfile(f'gwr_all_building_df.shape: {gwr_all_building_df.shape}, EGID: {gwr_all_building_df["EGID"].nunique()}', summary_file_name, 2, True)
    checkpoint_to_logfile(f'\t> selection range BFS municipalities: {bfs_number_def}', summary_file_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_file_name)
    checkpoint_to_logfile(f'after GSTAT selection, gwr.shape: {gwr_mrg1.shape} EGID.nunique: {gwr_mrg1["EGID"].nunique()} ({round((gwr_mrg1.shape[0] ) / gwr_mrg0.shape[0] * 100, 2)} % of gwr_mrg0)', summary_file_name, 2, True) 
    checkpoint_to_logfile(f'\t> selection GSTAT: {gwr_selection_specs_def["GSTAT"]} "only existing bulidings"', summary_file_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_file_name)
    checkpoint_to_logfile(f'after GKLAS selection, gwr.shape: {gwr_mrg2.shape} EGID.nunique: {gwr_mrg2["EGID"].nunique()} ({round((gwr_mrg2.shape[0] ) / gwr_mrg1.shape[0] * 100, 2)} % of gwr_mrg1)', summary_file_name, 2, True)
    checkpoint_to_logfile(f'\t> selection GKLAS: {gwr_selection_specs_def["GKLAS"]} "1110 - building w 1 living space, 1121 - w 2 living spaces, 1276 - agricluture buildings (stables, barns )"', summary_file_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_file_name)
    checkpoint_to_logfile(f'after GBAUJ_minmax selection, gwr.shape: {gwr_mrg3.shape} EGID.nunique: {gwr_mrg3["EGID"].nunique()} ({round((gwr_mrg3.shape[0] ) / gwr_mrg2.shape[0] * 100, 2)} % of gwr_mrg3)', summary_file_name, 2, True)
    checkpoint_to_logfile(f'\t> selection GBAUJ_minmax: {gwr_selection_specs_def["GBAUJ_minmax"]} "built construction between years"', summary_file_name, 2, True)
    # print_to_logfile(f'\n', log_file_name_def=summary_file_name)
    checkpoint_to_logfile(f'from ALL gwr_mrg0 (aggregated with dwelling, bfs already selected): {gwr_mrg0["EGID"].nunique()-gwr_mrg3["EGID"].nunique()} of {gwr_mrg0["EGID"].nunique()} EGIDs removed ({round((gwr_mrg0["EGID"].nunique() - gwr_mrg3["EGID"].nunique()  )/gwr_mrg0["EGID"].nunique()*100, 2)}%, mrg0-mrg3 of mrg0)', summary_file_name, 2, True)
    checkpoint_to_logfile(f'\t> {gwr_mrg3["EGID"].nunique()} gwr_mrg3 EGIDS {round((gwr_mrg3["EGID"].nunique())/gwr_mrg0["EGID"].nunique()*100, 2)}%  of  {gwr_mrg0["EGID"].nunique()} gwr_mrg0', summary_file_name, 2, True)
    print_to_logfile(f'\n', log_file_name_def=summary_file_name)


    # check proxy possiblity -------------------
    checkpoint_to_logfile(f'gwr_guilding_df.shape: {gwr_building_df.shape}, EGID: {gwr_building_df["EGID"].nunique()};\n  gwr_dwelling_df.shape: {gwr_dwelling_df.shape}, EGID: {gwr_dwelling_df["EGID"].nunique()};\n  gwr.shape: {gwr.shape}, EGID: {gwr["EGID"].nunique()}', log_file_name_def, 2, True)
    
    checkpoint_to_logfile(f'* check for WAZIM: {gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAZIM", {round((gwr.loc[gwr["WAZIM"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', log_file_name_def, 1, True)
    checkpoint_to_logfile(f'* check for WAREA: {gwr.loc[gwr["WAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty WAREA", {round((gwr.loc[gwr["WAREA"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', log_file_name_def, 1, True)
    checkpoint_to_logfile(f'* check for GAREA: {gwr.loc[gwr["GAREA"] != "", "EGID"].nunique()} unique EGIDs of non-empty GAREA", {round((gwr.loc[gwr["GAREA"] != "", "EGID"].nunique() ) / gwr["EGID"].nunique() * 100, 2)} % of total unique EGIDs ({gwr["EGID"].nunique()})', log_file_name_def, 1, True)

    # checkpoint_to_logfile('Did NOT us a combination of building and dwelling data, \n because they overlap way too little. This makes sense \nintuitievly as single unit houses probably are not registered \nas dwellings in the data base.', log_file_name_def, 1, True)


    # merge dfs and export -------------------
    # gwr = gwr_building_df
    gwr.to_csv(f'{data_path_def}/output/preprep_data/gwr.csv', sep=';', index=False)
    gwr_mrg_all_building_in_bfs.to_csv(f'{data_path_def}/output/preprep_data/gwr_mrg_all_building_in_bfs.csv', sep=';', index=False)
    gwr.to_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
    gwr_mrg_all_building_in_bfs.to_parquet(f'{data_path_def}/output/preprep_data/gwr_mrg_all_building_in_bfs.parquet')
    checkpoint_to_logfile(f'exported gwr data', log_file_name_def=log_file_name_def, n_tabs_def = 4)


    # create spatial df and export -------------------
    gwr = gwr.loc[(gwr['GKODE'] != '') & (gwr['GKODN'] != '')]
    gwr[['GKODE', 'GKODN']] = gwr[['GKODE', 'GKODN']].astype(float)
    gwr['geometry'] = gwr.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
    gwr_gdf = gpd.GeoDataFrame(gwr, geometry='geometry')
    gwr_gdf.crs = 'EPSG:2056'
    gwr_gdf = gwr_gdf.loc[:, ['EGID', 'geometry']]
    gwr_gdf.to_file(f'{data_path_def}/split_data_geometry/gwr_gdf.geojson', driver='GeoJSON')


