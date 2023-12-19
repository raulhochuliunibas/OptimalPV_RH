import os
import pandas as pd
import geopandas as gpd
import datetime

from datetime import datetime

wd_path = 'C:/Models/OptimalPV_RH'
data_path = f'{wd_path}_data'
os.chdir(wd_path)
os.listdir(data_path)

# create a export txt file for summary outputs
export_txt_name = f'{data_path}/sanitycheck_joined_roof_kat_pv_gm/sanity_check_output.txt'
with open(export_txt_name, 'w') as export_txt:
    export_txt.write(f'\n')
    export_txt.write(f'\n *************************** \n SANITY CHECK OUTPUT \n *************************** \n')
    export_txt.write(f'\n* start script: time: {datetime.now()}')


# import data from input
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()

with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* imported all shapes from input: time: {datetime.now()}')

# import data from parquet
gm_shp_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gmgm_shp.parquet')   
roof_kat_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/roof_kat.parquet')
bldng_reg_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/bldng_reg.parquet')
heatcool_dem_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/heatcool_dem.parquet')
pv_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/pv.parquet')

with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* imported all shapes from parquet: time: {datetime.now()}')

# import aggregates 
gm_shp_agg = pd.read_parquet(f'{data_path}/agg_sol_kat_pv_BY_GM/agg_solkat_pv_gm_


# import final aggregated data set



