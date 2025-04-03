import sys
import os
import pandas as pd
import geopandas as gpd
import requests
import copy

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from api_keys.api_keys import get_pvtarif_key

print(get_pvtarif_key())

# ------------------------------------------------------------------------------------------------------
# API DATA IMPORT
#> https://www.vese.ch/wp-content/uploads/pvtarif/pvtarif2/appPvMapExpert/pvtarif-map-expert-data-de.html
# ------------------------------------------------------------------------------------------------------


# MUNICIPALITY QUERY ------------------------------------------------------
def api_pvtarif_gm_ewr_Mapping(scen, ):

    '''
    This function imports DSO ID for all the selected BFS municipality numbers where they are operating. Keep unique ones per municipality, so to know which DSO operates where.  
    The data is saved as parquet file in the data folder.
    ''' 

    # SETUP --------------------------------------
    print_to_logfile('run function: api_pvtarif_gm_ewr_Mapping.py', scen.log_name)
    os.makedirs(f'{scen.data_path}/input_api', exist_ok=True)

    # QUERY --------------------------------------
    # gm_shp_df = gpd.read_file(f'{scen.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGRENZE.shp')
    gm_shp_df = gpd.read_file(f'{scen.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    

    bfs_list = gm_shp_df['BFS_NUMMER'].unique()
    bfs_counter = len(bfs_list) // 4

    url = 'https://opendata.vese.ch/pvtarif/api/getData/muni'
    response_bfs_list = []
    Map_df = []
    checkpoint_to_logfile('api call pvtarif gm to ewr started', scen.log_name, 2,scen.show_debug_prints)

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
                checkpoint_to_logfile(f'bfs: {bfs}, {i_bfs+1} of {len(bfs_list)} in list', scen.log_name, 3, scen.show_debug_prints)

    Map_gm_ewr = pd.concat(Map_df, ignore_index=True)
    checkpoint_to_logfile('api call pvtarif gm completed', scen.log_name, 3, scen.show_debug_prints)
    print_to_logfile(f'\n', scen.log_name)

    # EXPORT --------------------------------------
    Map_gm_ewr.to_parquet(f'{scen.data_path}/input_api/Map_gm_ewr.parquet')
    checkpoint_to_logfile('exported Map_gm_ewr from API', scen.log_name, 3)

    with open(f'{scen.data_path}/input_api/time_stamp.txt', 'w') as f:
        f.write(f'API call was run on : {pd.Timestamp.now()}')


def api_pvtarif_data(scen, ):
    '''
    This function imports the ID of all Distribution System Grid Operators of the VESE API (nrElcom) and their PV compensation tariff.
    The data is aggregated by DSO and year and saved as parquet file in the data folder.
    '''
    # SETUP --------------------------------------
    print_to_logfile('run function: api_pvtarif.py', scen.log_name)
    os.makedirs(f'{scen.data_path}/input_api', exist_ok=True)

    # QUERY --------------------------------------
    year_range_list = [str(year % 100).zfill(2) for year in range(scen.year_range[0], scen.year_range[1]+1)]
    response_all_df_list = []

    Map_gm_ewr = pd.read_parquet(f'{scen.data_path}/input_api/Map_gm_ewr.parquet')
    ew_id = Map_gm_ewr['nrElcom'].unique()

    ew_id_counter = len(ew_id) / 4

    url = "https://opendata.vese.ch/pvtarif/api/getData/evu?"

    for y in year_range_list:
        checkpoint_to_logfile(f'start api call pvtarif for year: {y}', scen.log_name, 3, scen.show_debug_prints)
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
        checkpoint_to_logfile(f'call year: {y} completed', scen.log_name, 3, scen.show_debug_prints)

    pvtarif_raw = pd.concat(response_all_df_list)
    checkpoint_to_logfile('api call pvtarif completed', scen.log_name, 3, scen.show_debug_prints)

    pvtarif = copy.deepcopy(pvtarif_raw)
    empty_cols = [col for col in pvtarif.columns if (pvtarif[col]=='').all()]
    pvtarif = pvtarif.drop(columns=empty_cols)

    # EXPORT --------------------------------------
    pvtarif.to_parquet(f'{scen.data_path}/input_api/pvtarif.parquet')
    checkpoint_to_logfile('exported pvtarif from API', scen.log_name, 3)

