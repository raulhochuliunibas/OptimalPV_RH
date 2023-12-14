import os as os
import geopandas as gpd

from function_data_import_aggregation import import_aggregate_data

script_run_on_server = 0

if script_run_on_server == 0:
    gm_shp = gpd.read_file(
    'C:/Models/OptimalPV_RH_data/input/swissboundaries3d_2023-01_2056_5728.shp', 
    layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
elif script_run_on_server == 1:
    gm_shp = gpd.read_file(
    'D:\RaulHochuli_inuse\OptimalPV_RH_data\input\swissboundaries3d_2023-01_2056_5728.shp',
    layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')


kt_list = list(gm_shp['KANTONSNUM'].dropna().unique())

for kt_i in kt_list:
    print('Kanton: ', str(int(kt_i)))
    gm_number_aggdef = list(gm_shp.loc[gm_shp['KANTONSNUM'] == kt_i, 'BFS_NUMMER'].unique())
    import_aggregate_data(
        name_aggdef = f'agg_kt_{str(int(kt_i))}' + str(kt_i), 
        script_run_on_server = script_run_on_server , 
        gm_number_aggdef = gm_number_aggdef
    )




