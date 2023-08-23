import os as os
import functions
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
import winsound

from functions import checkpoint_to_logfile
from datetime import datetime
from shapely.ops import unary_union

# still uncertain if this is needed
import warnings

# ----------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------


# pre setup + working directory -------------------------------------------------------------

wd_path = "C:/Models/OptimalPV_RH"
data_path = "C:/Models/OptimalPV_RH_data"
os.chdir(wd_path)   
# create log file for checkpoint comments
with open(f'log_file.txt', 'w') as log_file:
        log_file.write(f' \n')
chapter_to_logfile('started running main_file.py')

os.listdir(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen')



# import data ----------------------------------------------------------

# load administrative shapes
kt_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
gm_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
checkpoint_to_logfile(f'finished loading administrative shapes')

# load solar kataster shapes
roof_kat = gpd.read_file(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
checkpoint_to_logfile(f'finished loading solar kataster shapes')

# load pv installation points
pv = gpd.read_file(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
pv.head()
checkpoint_to_logfile(f'finished loading pv installation points')


# set crs to EPSG 4326 ----------------------------------------------------------

kt_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
gm_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
roof_kat.set_crs("EPSG:4326", allow_override=True, inplace=True)
checkpoint_to_logfile(f'finished setting crs to EPSG 4326')




# ----------------------------------------------------------------------
# Aggregate roof parts at house level 
# ----------------------------------------------------------------------
chapter_to_logfile('start aggr. roof parts at house level')


# subset to relevant houses ----------------------------------------------------------
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
roof_kat_sub = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
# export subset of relevant roof kat 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    roof_kat_sub.to_file(f'{data_path}/roof_kat2.shp', driver='ESRI Shapefile')


# create empty nan df pd_union with unique sb_obj_uuids as index ----------------------------------------------------------
sb_obj_unique = roof_kat_sub['SB_UUID'].unique() 
pd_union = pd.DataFrame(index = sb_obj_unique, columns = ['polygon_geom'])
cols = ['FLAECHE', 'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG']
cats = ['cat2_', 'cat3_', 'cat4_', 'cat5_']
new_col = [cat + col for cat in cats for col in cols ]
pd_union[new_col] = np.nan
checkpoint_to_logfile(f'created empty df for iter over roof parts')


# loop ----------------------------------------------------------------------
# iterating over all SB_UUID, adding roof_kat data by individual house 
cutoff_roof_kat_area = [10,300] #TODO: add here values from the PV installation data set
for idx, row_srs in pd_union.iterrows():
    
    # add unified geometry
    row_srs['polygon_geom'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union
    
    # set boolean indicatros 
    bool_id_2 = (roof_kat_sub['KLASSE'].isin([1,2])) & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])
    bool_id_3 = (roof_kat_sub['KLASSE'].isin([3]))   & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])
    bool_id_4 = (roof_kat_sub['KLASSE'].isin([4]))   & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])
    bool_id_5 = (roof_kat_sub['KLASSE'].isin([5]))   & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])

    for col in cols: 
        pd_union.loc[idx, f'cat2_{col}'] = roof_kat_sub.loc[bool_id_2, f'{col}'].sum()
        pd_union.loc[idx, f'cat3_{col}'] = roof_kat_sub.loc[bool_id_3, f'{col}'].sum()
        pd_union.loc[idx, f'cat4_{col}'] = roof_kat_sub.loc[bool_id_4, f'{col}'].sum()
        pd_union.loc[idx, f'cat5_{col}'] = roof_kat_sub.loc[bool_id_5, f'{col}'].sum()

checkpoint_to_logfile(f'finished loop iter over roof parts')
winsound.Beep(840,  100)
winsound.Beep(840,  100)



# transform pd to gdf ----------------------------------------------------------
roof_union = gpd.GeoDataFrame(pd_union, geometry = 'polygon_geom', crs = roof_kat2.crs)
roof_union.to_file(f'{data_path}/roof_union.shp')
checkpoint_to_logfile(f'pd.df transformed to geo df + exported')



# END ----------------------------------------------------------------------
chapter_to_logfile('END of main_file.py')
winsound.Beep(200, 300)
winsound.Beep(38,  150)
winsound.Beep(200, 300)
winsound.Beep(38,  150)
winsound.Beep(200, 1000)

# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------

#> asdfasdf



