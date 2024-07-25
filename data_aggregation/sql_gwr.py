import sys
import os
import pandas as pd
import geopandas as gpd
import  sqlite3

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
    print_to_logfile('run function: sql_gwr_data.py', log_file_name_def=log_file_name_def)

    # query -------------------

    query_columns =  ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                     'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                     'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1']
    conn = sqlite3.connect(f'{data_path_def}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    query_columns_str = ', '.join(query_columns)
    query_bfs_numbers = ', '.join([str(i) for i in bfs_number_def])
    checkpoint_to_logfile(f'use GWR query cols: {query_columns}', log_file_name_def=log_file_name_def, n_tabs_def = 5)


    # cur.execute("SELECT EGID, GDEKT FROM building")
    cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')

    sqlrows = cur.fetchall()

    conn.close()
    checkpoint_to_logfile('sql query done', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # convert to df -------------------
    gwr = pd.DataFrame(sqlrows, columns=query_columns)

    
    # export -------------------
    gwr.to_parquet(f'{data_path_def}/output/preprep_data/gwr.parquet')
    gwr.to_csv(f'{data_path_def}/output/preprep_data/gwr.csv', sep=';', index=False)
    checkpoint_to_logfile(f'exported gwr data', log_file_name_def=log_file_name_def, n_tabs_def = 5)

    







        





