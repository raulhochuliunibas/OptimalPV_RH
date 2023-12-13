import os as os
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
import winsound
from datetime import datetime

print(f'start script: time: {datetime.now()}')
winsound.Beep(840,  100)
    
wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
data_path = f'{wd_path}_data'

os.listdir(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/')

# import municipality shapes
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

# import all electricity production plants (not just pv)
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
pv = elec_prod.loc[elec_prod['SubCategory'] == 'subcat_2']
elec_prod['SubCategory'].value_counts()
pv.shape

# adjust coordinate reference system for identical projection
pv.set_crs(epsg = 2056, inplace=True, allow_override=True)
gm_shp.set_crs(epsg = 2056, inplace=True, allow_override=True)

# index correction because ChatGPT told me so 
if 'index_left' in pv.columns:
    pv = pv.rename(columns={'index_left': 'left_index'})
if 'index_right' in gm_shp.columns:
    gm_shp = gm_shp.rename(columns={'index_right': 'right_index'})

print(f'start sjoin: time: {datetime.now()}')
pv_joined = gpd.sjoin(pv, gm_shp, how="left", predicate="within")
print(f'end sjoin, start export to shp: time: {datetime.now()}')
pv_joined.to_file(f'{data_path}/output/pv_joined_BFS.shp')
print(f'end export to shp: time: {datetime.now()}')

# beep to indicate end of script
winsound.Beep(840,  100)
winsound.Beep(840,  100)
winsound.Beep(840,  100)
