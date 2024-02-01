import sys
import os
import pandas as pd
import geopandas as gpd
import  sqlite3

sys.path.append('..')
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# SQL DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# GEBÄUDE- UND WOHNUNGSREGISTER --------------------------------------------
def sql_gwr_data(
        script_run_on_server_def = None,
        recreate_parquet_files_def = None,
        smaller_import_def = None,
        log_file_name_def = None,
        wd_path_def = None,
        data_path_def = None,
        show_debug_prints_def = None,
        ):        
    """
    Function to import data from the GWR database
    """ 


    # setup -------------------
    wd_path = "D:/RaulHochuli_inuse/Models/OptimalPV_RH" if script_run_on_server_def else "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'

    # create directory + log file
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')
    checkpoint_to_logfile('run function: sql_gwr_data.py', log_file_name_def=log_file_name_def, n_tabs_def = 5) 

    query_columns = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 'WAREA', 'WNART',
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
        n_rows = 10000
        gm_shp_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
        solkat_gdf = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = n_rows)
        gwr_json = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson', rows=n_rows)
        
        

        # check GWR
        gwr_sql = gwr.copy()
        gwr_sql['EGID'] = pd.to_numeric(gwr_sql['EGID'], errors='coerce')   

        gwr_sql = gwr_sql.loc[gwr_sql['EGID'].isin(gwr_json['egid'])]
        gwr_sql['GKODE'] = pd.to_numeric(gwr_sql['GKODE'], errors='coerce')
        gwr_sql['GKODN'] = pd.to_numeric(gwr_sql['GKODN'], errors='coerce')
        gwr_sql_gdf = gpd.GeoDataFrame(gwr_sql, geometry=gpd.points_from_xy(gwr_sql['GKODE'], gwr_sql['GKODN']))

        # check solkat
        gwr_sql_2 = gwr.copy()
        gwr_sql_2['EGID'] = pd.to_numeric(gwr_sql_2['EGID'], errors='coerce')

        gwr_sql_2 = gwr_sql_2.loc[gwr_sql_2['EGID'].isin(solkat_gdf['GWR_EGID'])]
        gwr_sql_2['GKODE'] = pd.to_numeric(gwr_sql_2['GKODE'], errors='coerce')
        gwr_sql_2['GKODN'] = pd.to_numeric(gwr_sql_2['GKODN'], errors='coerce')
        gwr_sql_gdf_2 = gpd.GeoDataFrame(gwr_sql_2, geometry=gpd.points_from_xy(gwr_sql_2['GKODE'], gwr_sql_2['GKODN']))

        for col in gwr_sql_gdf_2.columns:
            if gwr_sql_gdf_2[col].dtype == 'datetime64[ns]':
                gwr_sql_gdf_2[col] = gwr_sql_gdf_2[col].astype(str)
        
        cols = ['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']
        solkat_gdf[cols] = solkat_gdf[cols].astype(str)

                
        gwr_json.set_crs(gm_shp_gdf.crs, inplace=True, allow_override=True)
        gwr_sql_gdf.set_crs(gm_shp_gdf.crs, inplace=True, allow_override=True)

        # solkat_gdf.set_crs(gm_shp_gdf.crs, inplace=True, allow_override=True)
        # gwr_sql_gdf_2.set_crs(gm_shp_gdf.crs, inplace=True, allow_override=True)
        
        gwr_json.to_file(f'{data_path}/output/check_gwr/gwr_json_{n_rows}.shp')
        solkat_gdf.to_file(f'{data_path}/output/check_gwr/solkat_{n_rows}.shp')
        gwr_sql_gdf.to_file(f'{data_path}/output/check_gwr/gwr_sql_{n_rows}.shp')
        gwr_sql_gdf_2.to_file(f'{data_path}/output/check_gwr/gwr_sql_2_{n_rows}.shp')
        

    if False: 

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








        





