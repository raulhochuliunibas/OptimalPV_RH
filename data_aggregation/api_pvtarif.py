import sys
import os
import pandas as pd
import geopandas as gpd
import requests

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

sys.path.append(r'C:/Models/OptimalPV_RH_apikeys')
from apikeys import get_pvtarif_key

print(get_pvtarif_key())

# ------------------------------------------------------------------------------------------------------
# API DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# PV TARIF VESE ------------------------------------------------------
#> https://www.vese.ch/wp-content/uploads/pvtarif/pvtarif2/appPvMapExpert/pvtarif-map-expert-data-de.html

def api_pvtarif_data(
        dataagg_settings_def, ):
    '''
    This function imports the ID of all Distribution System Grid Operators of the VESE API (nrElcom) and their PV compensation tariff.
    The data is aggregated by DSO and year and saved as parquet file in the data folder.
    ''' 

    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']
    print_to_logfile(f'run function: api_pvtarif.py', log_file_name_def=log_file_name_def)


    # query -------------------
    year_range_list = [str(year % 100).zfill(2) for year in range(year_range_def[0], year_range_def[1]+1)]
    response_all_df_list = []

    Map_gm_ewr = pd.read_parquet(f'{data_path_def}/output/preprep_data/Map_gm_ewr.parquet')
    ew_id = Map_gm_ewr['nrElcom'].unique()[0:20] if smaller_import_def else Map_gm_ewr['nrElcom'].unique()   

    # ew_id = list(range(1, 50+1)) if smaller_import_def else range(1, 1000+1)   # the ew ID is quite random in the range 1:1000. So I just loop over all of them and just keep the valid API calls

    ew_id_counter = len(ew_id) / 4

    url = "https://opendata.vese.ch/pvtarif/api/getData/evu?"

    for y in year_range_list:
        checkpoint_to_logfile(f'api call pvtarif for year: {y} started', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)
        response_ew_list = []

        for i_ew, ew in enumerate(ew_id):
            req_url = f'{url}evuId={ew}&year={y}&licenseKey={get_pvtarif_key()}'
            response = requests.get(req_url)
            response_json = response.json()

            if response_json['valid'] == True:
                response_ew_list.append(response_json)

            if i_ew % ew_id_counter == 0:
                checkpoint_to_logfile(f'-- year: {y}, ew: {ew}, {i_ew+1} of {len(ew_id)} in list', log_file_name_def=log_file_name_def, n_tabs_def = 2, show_debug_prints_def=show_debug_prints_def)

        response_ew_df = pd.DataFrame(response_ew_list)
        response_ew_df['year'] = y
        response_all_df_list.append(response_ew_df)
        checkpoint_to_logfile(f'api call year: {y} completed', log_file_name_def=log_file_name_def, n_tabs_def = 4, show_debug_prints_def=show_debug_prints_def)

    pvtarif_raw = pd.concat(response_all_df_list)

    # output transformations
    # pvtarif = pvtarif_raw.loc[pvtarif_raw['valid'] == True].copy()

    # remove all columns that have only na values
    pvtarif = pvtarif_raw.copy()
    empty_cols = [col for col in pvtarif.columns if (pvtarif[col]=='').all()]
    pvtarif = pvtarif.drop(columns=empty_cols)

    # export
    pvtarif.to_parquet(f'{data_path_def}/split_data_geometry/pvtarif.parquet')
    pvtarif.to_csv(f'{data_path_def}/split_data_geometry/pvtarif.csv', index=False)
    checkpoint_to_logfile(f'exported electricity prices', log_file_name_def=log_file_name_def, n_tabs_def = 5)


    

# MUNICIPALITY QUERY ------------------------------------------------------
def api_pvtarif_gm_ewr_Mapping(
        dataagg_settings_def, ):

    '''
    This function imports DSO ID for all the selected BFS municipality numbers where they are operating. Keep unique ones per municipality, so to know which DSO operates where.  
    The data is saved as parquet file in the data folder.
    ''' 

    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']
    print_to_logfile(f'run function: api_pvtarif_gm_Mapping.py', log_file_name_def=log_file_name_def)


    # query -------------------
    gm_shp = gpd.read_file(f'{data_path_def}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    bfs_list = gm_shp['BFS_NUMMER'].unique()

    # bfs_list = bfs_number_def 
    bfs_counter = len(bfs_list) // 4

    url = 'https://opendata.vese.ch/pvtarif/api/getData/muni'
    response_bfs_list = []
    Map_df = []
    checkpoint_to_logfile(f'api call pvtarif gm to ewr started', log_file_name_def=log_file_name_def, n_tabs_def = 2, show_debug_prints_def=show_debug_prints_def)

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
                checkpoint_to_logfile(f'bfs: {bfs}, {i_bfs+1} of {len(bfs_list)} in list', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)

    Map_gm_ewr = pd.concat(Map_df, ignore_index=True)   
    checkpoint_to_logfile(f'api call pvtarif gm completed', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)
    print_to_logfile(f'\n', log_file_name_def=log_file_name_def)

    # export   
    Map_gm_ewr.to_parquet(f'{data_path_def}/output/preprep_data/Map_gm_ewr.parquet')
    Map_gm_ewr.to_csv(f'{data_path_def}/output/preprep_data/Map_gm_ewr.csv', index=False)
    checkpoint_to_logfile(f'exported Map_gm_ewr from API', log_file_name_def=log_file_name_def, n_tabs_def = 3)

    # NOTE: Is this still necessary? following script should pick up data from preprep_data folder
    Map_gm_ewr.to_parquet(f'{data_path_def}/split_data_geometry/Map_gm_ewr.parquet')



