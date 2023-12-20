import os as os
import geopandas as gpd
import glob
import shutil
import winsound

from local_data_import_aggregation import import_aggregate_data

# SETTIGNS --------------------------------------------------------------------
script_run_on_server = 0


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


# AGGREGATION -----------------------------------------------------------------

# export for all of CH
if True:
    import_aggregate_data(
        name_aggdef = 'agg_solkat_pv_gm_ALL_CH', 
        script_run_on_server = script_run_on_server , 
        data_source= 'parquet'
    )


# export aggregations by cantons 
if False:
    kt_list = list(gm_shp['KANTONSNUM'].dropna().unique())

    for kt_i in kt_list:
        gm_number_aggdef = list(gm_shp.loc[gm_shp['KANTONSNUM'] == kt_i, 'BFS_NUMMER'].unique())
        print(f'\n\n ***** Kanton:{str(int(kt_i))} *****')
        # print(f'> municipality numbers:{gm_number_aggdef}')
        
        import_aggregate_data(
            name_aggdef = f'agg_solkat_pv_gm_KT{str(int(kt_i))}', 
            script_run_on_server = script_run_on_server , 
            gm_number_aggdef = gm_number_aggdef, 
            data_source= 'parquet')
        
    # copy all subfolders to one folder
    if not os.path.exists(f'{data_path}/agg_sol_kat_pv_BY_KT'):
        os.makedirs(f'{data_path}/agg_sol_kat_pv_BY_KT')
    source_dir = glob.glob(f'{data_path}/*agg_solkat_pv_gm_KT*')
    target_dir = f'{data_path}/agg_sol_kat_pv_BY_KT'
    # dirs = glob.glob(os.path.join(source_dir, '*agg_solkat_pv_gm_*'))
    for f in source_dir:
        shutil.move(f, target_dir)

    # create a folder containing all parquet and log files
    if  not os.path.exists(f'{data_path}/agg_sol_kat_pv_BY_KT/agg_solkat_pv_gm_ALL'):
        os.makedirs(f'{data_path}/agg_sol_kat_pv_BY_KT/agg_solkat_pv_gm_ALL')
    
    # add parquet and log files
    files_pq = glob.glob(f'{data_path}/agg_sol_kat_pv_BY_KT/*/*agg_solkat_pv_gm_KT*.parquet')
    files_txt = glob.glob(f'{data_path}/agg_sol_kat_pv_BY_KT/*/*agg_solkat_pv_gm*.txt')
    files = files_pq + files_txt
    for f in files:
        shutil.copy(f, f'{data_path}/agg_sol_kat_pv_BY_KT/agg_solkat_pv_gm_ALL')    

    # remove unnecessary files
    files_del = glob.glob(f'{data_path}/agg_sol_kat_pv_BY_KT/agg_solkat_pv_gm_ALL/agg_solkat_pv_gm_*_selected_gm_shp.parquet')
    for f in files_del:
        os.remove(f)

