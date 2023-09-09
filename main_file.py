import os as os
import functions
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
import winsound

from functions import chapter_to_logfile, checkpoint_to_logfile
from datetime import datetime
from shapely.ops import unary_union

# still uncertain if this is needed
import warnings

# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------
wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
#wd_path = "D:\OptimalPV_RH"         # path for server directory
# poetry add pandas numpy geopandas matplotlib pyogrio shapely


# pre setup + working directory -----------------------------------------------------------------------------------
winsound.Beep(840,  100)

data_path = f'{wd_path}_data'
os.chdir(wd_path)   
# create log file for checkpoint comments
with open(f'log_file.txt', 'w') as log_file:
        log_file.write(f' \n')
chapter_to_logfile('started running main_file.py')

os.listdir(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen')


# import data -----------------------------------------------------------------------------------------------------

# load administrative shapes
kt_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
#kt_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
gm_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
checkpoint_to_logfile(f'finished loading administrative shapes')

# load solar kataster shapes
roof_kat = gpd.read_file(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
checkpoint_to_logfile(f'finished loading roof solar kataster shapes')
faca_kat = gpd.read_file(f'{data_path}/solarenergie-eignung-fassaden_2056.gdb/SOLKAT_FASS_20230221.gdb', layer ='SOLKAT_CH_FASS') 
checkpoint_to_logfile(f'finished loading facade solar kataster shapes')

# load building register indicating residential or industrial use
bldng_reg = gpd.read_file(f'{data_path}/GebWohnRegister.CH/buildings.geojson')
checkpoint_to_logfile(f'finished loading building register points')

# load heating / cooling demand raster 150x150m
heatcool_dem = gpd.read_file(f'{data_path}/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
checkpoint_to_logfile(f'finished loading heating and cooling demand points')

# load pv installation points
pv = gpd.read_file(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
checkpoint_to_logfile(f'finished loading pv installation points')

# check if all CRS are compatible
kt_shpo.crs == gm_shp.crs == roof_kat.crs == faca_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs





# ----------------------------------------------------------------------------------------------------------------
# Aggregate roof parts at house level 
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('aggregate roof parts at house level')
winsound.Beep(840,  100)
winsound.Beep(840,  100)


# set aggregate parameters  ---------------------------------------------------------------------------------------

# subset to relevant houses 
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
#cat_sb_object = [4, 13]#[1,2,4,5,8,12,13]
#roof_kat['SB_OBJEKTART'].value_counts()
#roof_kat_sub = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()

roof_kat_sub = roof_kat.loc[roof_kat['SB_UUID'].isin(roof_kat['SB_UUID'].unique()[0:1000])].copy()


# define roof range which are considered
cutoff_roof_kat_area = [10,300] #TODO: add here values from the PV installation data set


# roof kataster: create empty nan gdf with unique sb_obj_uuids as index --------------------------------------------
sb_obj_unique = roof_kat_sub['SB_UUID'].unique() 
roof_union = gpd.GeoDataFrame(index = sb_obj_unique, columns = ['geometry']) #TODO: find a better naming convetion!
roof_union.set_crs(roof_kat.crs, inplace=True)

cols = ['FLAECHE', 'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG']
cats = ['cat2_', 'cat3_', 'cat4_', 'cat5_']
new_col = [cat + col for cat in cats for col in cols ]
roof_union[new_col] = np.nan
checkpoint_to_logfile(f'created empty df to then iter over roof parts')


# roof kataster: loop over roof_kat  --------------------------------------------------------------------------------
idx = '{AF378B63-B28F-4A92-9BEB-4B84ABD75BDF}' #TODO: delete later when no longer used
set_buffer = 3 # determines the buffer around shapes to ensure a more proper union merge of single ouse shapes
for idx, row_srs in roof_union.iterrows():
    
    # add unified geometry
    #row_srs['geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union
    #roof_union.loc[idx, 'geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union

    # tester
    roof_union.loc[idx, 'geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16) # roof_geom_buff.buffer(-0.5, resolution = 16).copy()
    #

    """
    # set boolean indicatros 
    bool_id_2 = (roof_kat_sub['KLASSE'].isin([1,2])) & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])
    bool_id_3 = (roof_kat_sub['KLASSE'].isin([3]))   & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])
    bool_id_4 = (roof_kat_sub['KLASSE'].isin([4]))   & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])
    bool_id_5 = (roof_kat_sub['KLASSE'].isin([5]))   & (roof_kat_sub['SB_UUID'] == idx ) & (roof_kat_sub['FLAECHE'] > cutoff_roof_kat_area[0]) & (roof_kat_sub['FLAECHE'] < cutoff_roof_kat_area[1])

    # add roof part values to aggr df
    for col in cols: 
        roof_union.loc[idx, f'cat2_{col}'] = roof_kat_sub.loc[bool_id_2, f'{col}'].sum()
        roof_union.loc[idx, f'cat3_{col}'] = roof_kat_sub.loc[bool_id_3, f'{col}'].sum()
        roof_union.loc[idx, f'cat4_{col}'] = roof_kat_sub.loc[bool_id_4, f'{col}'].sum()
        roof_union.loc[idx, f'cat5_{col}'] = roof_kat_sub.loc[bool_id_5, f'{col}'].sum()
    """

checkpoint_to_logfile(f'finished loop iter over roof parts')
roof_union.to_file(f'{data_path}/z_py_exports/roof_union_W{set_buffer}buffer.shp')


# pv inst: add empty nan conlumns to aggr df   ----------------------------------------------------------------------

roof_union['pv_d', 'pv_address', 'pv_postcode', 'pv_BeginningO', 'InitialPow', 'TotalPow' ] = np.nan


# pv inst: loop over pv   -------------------------------------------------------------------------------------------

for idx, row_srs in roof_union.iterrows():

    bool_id_pvonroof = pv['geometry'].intersects(roof_union.loc[idx, 'geometry'].buffer(0, resolution = 16))
    roof_union.loc[idx, 'pv_d'] = bool_id_pvonroof.sum()
    """
    roof_union.loc[idx, 'pv_address'] 
    roof_union.loc[idx, 'pv_postcode']
    roof_union.loc[idx, 'pv_BeginningO']
    roof_union.loc[idx, 'InitialPow']
    roof_union.loc[idx, 'TotalPow']
    """


# export 
roof_union.to_file(f'{data_path}/z_py_exports/roof_union.shp')
roof_union.to_file(f'{data_path}/z_py_exports/roof_union.geojson', driver='GeoJSON')
checkpoint_to_logfile(f'aggregated gdf exported')


# --------------------------------------------------------------------------------------------------------------------
# END 
# --------------------------------------------------------------------------------------------------------------------
chapter_to_logfile('END of main_file.py')
winsound.Beep(400, 100)
winsound.Beep(400, 100)
winsound.Beep(400, 500)

# --------------------------------------------------------------------------------------------------------------------
# book mark ---------------------------------------------------------------------------------------------------------- 
# --------------------------------------------------------------------------------------------------------------------


# TODO: CRS is problematic! after changing it to 4326, the geometry cannot be plotted anymore. 



# ----------------------------------------------------------------------


roof_kat_sub = roof_kat.loc[roof_kat['SB_UUID'].isin(roof_kat['SB_UUID'].unique()[0:1000])].copy()
roof_kat_sub['geometry'] = roof_kat_sub['geometry'].copy().buffer(0.5, resolution = 16)
roof_kat_sub.to_file(f'{data_path}/z_py_exports/roof_kat_sub_buffer.shp')   

roof_kat_sub['geometry'] = roof_kat_sub['geometry'].copy().buffer(-0.5, resolution = 16)
roof_kat_sub.to_file(f'{data_path}/z_py_exports/roof_kat_sub_DEbuffer.shp')   




type(gm_shp)
gm_shp.to_file(f'{data_path}/gm_shp.shp')
gm_shp.to_file(f'{data_path}/gm_shp.geojson', driver='GeoJSON')

type(roof_kat_sub)
roof_kat_sub.to_file(f'{data_path}/z_py_exports/roof_kat_sub.geojson', driver='GeoJSON')


type(roof_union)
roof_union.crs
roof_union.head()
roof_union.info()
roof_union['geometry'].value_counts()
roof_union['geometry'].plot()
plt.show()




roof_union.columns
type(roof_union)
type(roof_union['geometry'])
roof_union ['geometry']
roof_union.loc[idx, 'geometry'].buffer(1, resolution = 16)
type(pv['geometry'])





kt_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
kt_shp.crs
kt_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)

kt_shp.plot()
plt.show()

gm_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
gm_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
