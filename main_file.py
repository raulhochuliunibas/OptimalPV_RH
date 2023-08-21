import pandas as pd
import numpy as np
import os as os
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio

from functions import crs2wsg84
from datetime import datetime
from shapely.ops import unary_union
#from pandasgui import show



# pre setup + working directory -------------------------------------------------------------
check = f'\n\n******************************\n started running main_file.py \n start at:{datetime.now()} \n******************************\n\n'
print(check)
with open(f'log_file.txt', 'w') as log_file:
    log_file.write(f'{check}\n')

wd_path = "C:/Models/OptimalPV_RH"
data_path = "C:/Models/OptimalPV_RH_data"
os.chdir(wd_path)   
os.listdir()
os.listdir(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen')



# import data ----------------------------------------------------------

# load administrative shapes
kt_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
gm_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
check = f'* finished loading administrative shapes: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")

# load solar kataster shapes
roof_kat = gpd.read_file(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
check = f'* finished loading roof kataster shapes: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")



# set crs to EPSG 4326 ----------------------------------------------------------

kt_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
gm_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
roof_kat.set_crs("EPSG:4326", allow_override=True, inplace=True)

check = f'\n* changed all CRS to EPSG 4326: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")


# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------
print("\n\n subset roof_kat for certain building types \n******************************\n\n")
"""
0 Bruecke gedeckt
1 Gebaeude Einzelhaus
2 Hochhaus
3 Hochkamin
4 Turm
5 Kuehlturm
6 Lagertank
7 Lueftungsschacht
8 Offenes Gebaeude
9 Treibhaus
10 Im Bau
11 Kapelle
12 Sakraler Turm
13 Sakrales Gebaeude
15 Flugdach
16 Unterirdisches Gebaeude
17 Mauer gross
18 Mauer gross gedeckt
19 Historische Baute
20 Gebaeude unsichtbar
"""
cat_sb_object = [2,4]#[1,2,4,5,8,12,13]
roof_kat['SB_OBJEKTART'].value_counts()
roof_kat2 = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()


roof_kat2.to_file(f'{data_path}/roof_kat2.shp')
roof_kat.columns
sb_obj_unique = roof_kat2['SB_UUID'].unique() 
type(sb_obj_unique)

pd_union = pd.DataFrame(index = sb_obj_unique, columns = ['polygon_geom'])

ctoff_roof_kat_area = [10,300] #TODO: add here values from the PV installation data set
for idx, n_row in pd_union.iterrows():
    # add unified geometry
    pd_union.loc[idx, 'polygon_geom'] = roof_kat2.loc[roof_kat2['SB_UUID'] == idx, 'geometry'].unary_union
    # add roof area cat_1to3
    

roof_union = gpd.GeoDataFrame(pd_union, geometry = 'polygon_geom', crs = roof_kat2.crs)
roof_union.to_file(f'{data_path}/roof_union.shp')

check = f'\n* created roof_union with unified geometriey + exported to shp: {datetime.now()}'
print(check)
with open(f'log_file.txt', 'a') as log_file:
    log_file.write(f"{check}\n")




print("\n\nfinished running main_file.py \n******************************\n\n")

# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------
roof_kat2.columns
asdf = sb_obj_unique[7]
roof_kat2.loc[roof_kat2['SB_UUID'] == asdf & roof_kat2[], 'geometry']
# add roof area cat_1to3
pd_union.loc[idx, 'roof_area'] = roof_kat2.loc[roof_kat2['SB_UUID'] == idx, 'FLAECHE'].sum()
