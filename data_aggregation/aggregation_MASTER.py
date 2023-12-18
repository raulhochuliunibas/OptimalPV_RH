import os as os
import geopandas as gpd
import winsound

from local_data_import_aggregation import import_aggregate_data

# SETUP & SETTIGNS --------------------------------------------------------------------
script_run_on_server = 0


# SETUP --------------------------------------------------------------------

if script_run_on_server == 0:
    gm_shp = gpd.read_file(
    'C:/Models/OptimalPV_RH_data/input/swissboundaries3d_2023-01_2056_5728.shp', 
    layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
elif script_run_on_server == 1:
    gm_shp = gpd.read_file(
    'D:\RaulHochuli_inuse\OptimalPV_RH_data\input\swissboundaries3d_2023-01_2056_5728.shp',
    layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')


# export aggregations by cantons
kt_list = list(gm_shp['KANTONSNUM'].dropna().unique())


for kt_i in kt_list:
    gm_number_aggdef = list(gm_shp.loc[gm_shp['KANTONSNUM'] == kt_i, 'BFS_NUMMER'].unique())
    print(f'\n ***** Kanton:{str(int(kt_i))} ***** \n')
    print(f'> municipality numbers:{gm_number_aggdef}')
    
    import_aggregate_data(
        name_aggdef = f'agg_solkat_pv_gm_kt_{str(int(kt_i))}', 
        script_run_on_server = script_run_on_server , 
        gm_number_aggdef = gm_number_aggdef, 
        select_solkat_aggdef = [1,2],)

# export aggreagtion for all gm
import_aggregate_data(
    name_aggdef = 'agg_all_gm', 
    script_run_on_server = script_run_on_server , 
    select_solkat_aggdef=[1,2],
)







