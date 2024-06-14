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
def sql_gwr_data(
        script_run_on_server_def = None,
        smaller_import_def = None,
        log_file_name_def = None,
        wd_path_def = None,
        data_path_def = None,
        show_debug_prints_def = None,
        ):        
    """
    Function to import data from the Building and Dwelling (Gebaeude und Wohungsregister) database.
    Import data from SQL file and save the relevant variables locally as parquet file.
    """ 


    # setup -------------------
    wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH" if script_run_on_server_def else "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'

    # create directory + log file
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')
    checkpoint_to_logfile('run function: sql_gwr_data.py', log_file_name_def=log_file_name_def, n_tabs_def = 5) 

    query_columns = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                     'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                     'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1']


    # query -------------------
    conn = sqlite3.connect(f'{data_path}/input/GebWohnRegister.CH/data.sqlite')
    cur = conn.cursor()
    query_columns_str = ', '.join(query_columns)

    # cur.execute("SELECT EGID, GDEKT FROM building")
    cur.execute(f'SELECT {query_columns_str} FROM building')

    sqlrows = cur.fetchall()

    conn.close()
    checkpoint_to_logfile('sql query done', log_file_name_def=log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    # convert to df -------------------
    gwr = pd.DataFrame(sqlrows, columns=query_columns)

    
    # export -------------------
    gwr.to_parquet(f'{data_path}/output/preprep_data/gwr.parquet')
    checkpoint_to_logfile(f'exported gwr data', log_file_name_def=log_file_name_def, n_tabs_def = 5)

    

    #------------------------------------------------------------------------------------------------------
    # GWR check
    #------------------------------------------------------------------------------------------------------

    if False: 
        """
        This is an old code snippet, initially used to check the data consistency of the GWR data. to the solkat data (solar potential by roof).  
        """
        # gwr_pq = pd.read_parquet(f'{data_path}/output/preprep_data_20240120/gwr_by_gm.parquet')
        solkat = pd.read_parquet(f'{data_path}/output/preprep_data_20240120/solkat_by_gm.parquet')

        gwr = gwr.loc[gwr['GABBJ'] == ""]

        # see which df has more unique egid
        gwr_egid = gwr['EGID'].unique()
        gwr_egid_n = len(gwr_egid)
        solkat_egid = solkat['GWR_EGID'].unique()
        solkat_egid_n = len(solkat_egid)

        (gwr_egid_n - solkat_egid_n) / gwr_egid_n

        sbuuid = solkat['SB_UUID'].unique()
        sbuuid_n = len(sbuuid)


        solkat_eg = solkat['GWR_EGID'].unique()
        solkat_eg_n = len(solkat_eg)

        (sbuuid_n - solkat_eg_n) / sbuuid_n


        # compare how many of solkat egid are in gwr

        solkat_egid = solkat['GWR_EGID'].unique()
        solkat_egid_n = len(solkat_egid)








        





