import os 
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union


script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server

# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------


# pre setup + working directory ----------------------------------------------------------------------------------

# set working directory
if script_run_on_server == 0:
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\RaulHochuli_inuse\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)


# import data sets -----------------------------------------------------------------------------------------------
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
print(f'*imported gm_shp')
roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
print(f'*imported roof_kat')
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
print(f'*imported elec_prod')
pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
print(f'*imported pv')   


# align CRSs
roof_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
pv.set_crs(gm_shp.crs, allow_override=True, inplace=True)
gm_shp.crs == roof_kat.crs == pv.crs

# take only residential houses for now
cat_sb_object = [1,]
roof_kat = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
print(f'*subset_roof_kat')

# unionize buffered polygons
set_buffer = 1.25
roof_agg = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
print(f'*unionized roof_kat')

# intersection of all 3 data sets --------------------------------------------------------------------------------

df2_roof_pv = gpd.sjoin(roof_agg, pv, how = "left", predicate = "intersects")
df3_joined_roof_pv_munic = gpd.sjoin(df2_roof_pv, gm_shp, how = "left", predicate = "intersects")   

df3_joined_roof_pv_munic.to_file(f'{data_path}/joined_roof_pv_munic.shp')

print(f'\n\n\n ************\nscript finished\n************\n\n\n')
