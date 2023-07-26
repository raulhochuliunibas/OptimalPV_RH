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
wd_path = "C:/Models/OptimalPV"
data_path = "C:/Models/data"
os.chdir(wd_path)   
os.listdir()
os.listdir(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen')



# import data ----------------------------------------------------------

# load administrative shapes
ch_kt = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
wgs84_crs = ch_kt.crs.to_string().split(" +up")[0]
ch_kt = ch_kt.to_crs(wgs84_crs)

ch_gm = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
wgs84_crs = ch_gm.crs.to_string().split(" +up")[0]
ch_gm = ch_gm.to_crs(wgs84_crs)

check = f'* loaded administrative shapes: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")

# load solar kataster shapes
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
shp_roof = ch_gm.unary_union



# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------


ch_kt_zh = ch_kt[ch_kt['NAME'] == 'Zürich']


asdf = gpd.sjoin(ch_gm, ch_kt_zh, how='inner', op='intersects')
type(asdf)




print("stuff acutally happend")
print("\n\nfinished running main_file.py \n******************************\n\n")
