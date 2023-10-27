import os as os
import pandas as pd
import numpy as np
import geopandas as gpd
import pyogrio
import winsound


wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
data_path = f'{wd_path}_data'

os.listdir(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp')


gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
elec_prod = gpd.read_file(f'{elec_prod_path}/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
pv = elec_prod.loc[elec_prod['SubCategory'] == 'subcat_2']
elec_prod['SubCategory'].value_counts()
pv.shape

pv.set_crs(epsg = 3857, inplace=True, allow_override=True)
gm_shp.set_crs(gm_shp.crs, inplace=True, allow_override=True)

if 'index_left' in pv.columns:
    pv = pv.rename(columns={'index_left': 'left_index'})
if 'index_right' in gm_shp.columns:
    gm_shp = gm_shp.rename(columns={'index_right': 'right_index'})

pv_joined = gpd.sjoin(pv, gm_shp, how="left", predicate="within")
pv_joined.to_file(f'{data_path}/output/pv_joined_BFS.shp')
print('finished export to shapefile')
