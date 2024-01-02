import os as os
import geopandas as gpd
import glob
import shutil
import winsound

from functions import chapter_to_logfile, checkpoint_to_logfile
from data_aggregation.local_data_import_aggregation import import_aggregate_data
from data_aggregation.spatial_data_toparquet_by_gm import spatial_toparquet

# SETTIGNS --------------------------------------------------------------------
script_run_on_server = 0
recreate_parquet_files = 0


# SETUP -----------------------------------------------------------------------
if script_run_on_server == 0:
    wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'
    
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp',
                           layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

elif script_run_on_server == 1:
    wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
    data_path = f'{wd_path}_data'

    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp',
                           layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    

# SPATIAL DATA TO PARQUET -----------------------------------------------------
pq_dir_exists = os.path.exists(f'{data_path}/spatial_intersection_by_gm')
pq_files_rerun = recreate_parquet_files == 1

if not pq_dir_exists or pq_files_rerun:
    spatial_toparquet(script_run_on_server_def = script_run_on_server)
    print('recreated parquet files for faster import and transformation')
else:
    print('parquet files exist already, no recreation necessary')


# AGGREGATIONS -----------------------------------------------------------------

# aggregation solkat, pv, munic, gwr, heatcool by cantons 
if True:
    kt_list = list(gm_shp['KANTONSNUM'].dropna().unique())  

    for n, kt_i in enumerate(kt_list):
        gm_number_aggdef = list(gm_shp.loc[gm_shp['KANTONSNUM'] == kt_i, 'BFS_NUMMER'].unique())
        import_aggregate_data(
            name_aggdef = f'agg_solkat_pv_gm_gwr_heat_KT{str(int(kt_i))}', 
            script_run_on_server = script_run_on_server , 
            gm_number_aggdef = gm_number_aggdef, 
            data_source= 'parquet')
        
        print(f'canton {kt_i} aggregated, {n+1} of {len(kt_list)} completed')
        
    # copy all subfolders to one folder
    name_dir_export ='agg_solkat_pv_gm_gwr_heat_BY_KT'
    if not os.path.exists(f'{data_path}/{name_dir_export}'):
        os.makedirs(f'{data_path}/{name_dir_export}')
    # source_dir = glob.glob(f'{data_path}/agg_solkat_pv_gm_gwr_heat_KT*')
    # target_dir = f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT'
    # # dirs = glob.glob(os.path.join(source_dir, '*agg_solkat_pv_gm_*'))
    # for f in source_dir:
    #     shutil.move(f, target_dir)

    # # create a folder containing all parquet and log files
    # if  not os.path.exists(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/agg_ALL'):
    #     os.makedirs(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/agg_ALL')
    # elif os.path.exists(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/agg_ALL'):
    #     os.remove(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/agg_ALL')
    
    # add parquet and log files
    files_copy = glob.glob(f'{data_path}/agg_solkat_pv_gm_gwr_heat_KT*/*agg_solkat_pv_gm_gwr_heat_KT*')
    # files_txt = glob.glob(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/*/*agg_solkat_pv_gm_gwr_heat_KT*.txt')
    # files = files_pq + files_txt
    for f in files_copy:
        shutil.move(f, f'{data_path}/{name_dir_export}')    

    # # remove unnecessary files
    # files_del = glob.glob(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/agg_ALL/*_selected_gm_shp.parquet')
    # for f in files_del:
    #     os.remove(f)
    files_del = glob.glob(f'{data_path}/agg_solkat_pv_gm_gwr_heat_KT*')
    for f in files_del:
        os.remove(f)

    if script_run_on_server == 0:
        winsound.Beep(2500, 1000)
        winsound.Beep(2500, 1000)

