import pandas as pd
import numpy as np
import os as os
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
from functions import crs2wsg84
from datetime import datetime
#from pandasgui import show


# pre setup -------------------------------------------------------------
check = f'\n\n******************************\n started running main_file.py \n start at:{datetime.now()} \n******************************\n\n'
print(check)
with open(f'log_file.txt', 'w') as log_file:
    log_file.write(f'{check}\n')

# set working directory -----------------------------------------------
wd_path = "C:/Models/OptimalPV_RH"
data_path = "C:/Models/OptimalPV_RH_data"
os.chdir(wd_path)   
os.listdir()
os.listdir(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen')



# import data ----------------------------------------------------------

# load administrative shapes
kt_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
wgs84_crs = kt_shp.crs.to_string().split(" +up")[0]
kt_shp = kt_shp.to_crs(wgs84_crs)

gm_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
wgs84_crs = gm_shp.crs.to_string().split(" +up")[0]
gm_shp = gm_shp.to_crs(wgs84_crs)

check = f'* finished loading administrative shapes: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")


# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------


# load solar kataster shapes
roof_kat = gpd.read_file(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
roof_kat.head()
kt_shp.head()
gm_shp.head()   
kt_shp.columns


kt_shp_be = kt_shp.loc[kt_shp["KANTONSNUM"] == 13].copy(deep=True)
kt_shp_be.plot(figsize=(10,10))
plt.show()
kt_shp_be.head()

roof_kat_be = gpd.sjoin(roof_kat, kt_shp_be, op='within')

"""
dat_solkat = gpd.read_file(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
wgs84_crs = dat_solkat.crs.to_string().split(" +up")[0]
dat_solkat = dat_solkat.to_crs(wgs84_crs)
"""

check = f'* loaded solar kataster shapes: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")

# load pv installations
"""
dat_pv = gpd.read_file(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen', layer ='subcat_2_pv')
wgs84_crs = dat_pv.crs.to_string().split(" +up")[0]
dat_pv = dat_pv.to_crs(wgs84_crs)
"""

check = f'* loaded solar kataster shapes: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")

# create roofs from kataster shapes ------------------------------------
shp_roof = gm_shp.unary_union



# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------


kt_shp_zh = kt_shp[kt_shp['NAME'] == 'Zürich']


asdf = gpd.sjoin(gm_shp, kt_shp_zh, how='inner', op='intersects')
type(asdf)




print("stuff acutally happend")
print("\n\nfinished running main_file.py \n******************************\n\n")
