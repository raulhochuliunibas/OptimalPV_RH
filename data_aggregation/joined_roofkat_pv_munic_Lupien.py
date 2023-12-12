import os 
import pandas as pd
import numpy as np
import geopandas as gpd
import winsound

from shapely.ops import unary_union
from datetime import datetime


script_run_on_server = 1         # 0 = script is running on laptop, 1 = script is running on server

# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------


# pre setup + working directory ----------------------------------------------------------------------------------

# set working directory
start_timer = datetime.now()

if script_run_on_server == 0:
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\RaulHochuli_inuse\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)


# import data sets -----------------------------------------------------------------------------------------------
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
print(f'*imported gm_shp | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', 
                         layer ='SOLKAT_CH_DACH')
# roof_kat['SB_UUID'].nunique()
# roof_kat['SB_UUID'].head()
# roof_kat['GWR_EGID'].nunique()
# roof_kat['GWR_EGID'].head(50)

print(f'*imported roof_kat | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
print(f'*imported elec_prod | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
print(f'*subset pv | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')

# align CRSs
roof_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
pv.set_crs(gm_shp.crs, allow_override=True, inplace=True)
gm_shp.crs == roof_kat.crs == pv.crs
print(f'*aligned CRSs | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')

# # take only residential houses for now
# cat_sb_object = [1,]
# roof_kat = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
# print(f'*subset_roof_kat')

# unionize buffered polygons
set_buffer = 1.25
roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
roof_agg = gpd.GeoDataFrame(roof_agg_Srs, geometry=roof_agg_Srs)
roof_agg.set_crs(gm_shp.crs, allow_override=True, inplace=True)
print(f'*unionized roof_kat | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')

# intersection of all 3 data sets --------------------------------------------------------------------------------
roof_kat.crs == gm_shp.crs == pv.crs == roof_agg.crs

df_join1 = gpd.sjoin(roof_agg, roof_kat, how = "left", predicate = "intersects")
df_join1.rename(columns={'index_right': 'index_roofkat'}, inplace=True)
print(f'*joined df1: roof_kat | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
df_join2 = gpd.sjoin(df_join1, pv, how = "left", predicate = "intersects")
df_join2.rename(columns={'index_right': 'index_pv'}, inplace=True)
print(f'*joined df2: pv | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
df_join3 = gpd.sjoin(df_join2, gm_shp, how = "left", predicate = "intersects")
df_join3.rename(columns={'index_right': 'index_gm'}, inplace=True)
print(f'*joined df3: gm_shp | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')



df_join3.info()
date_ts_cols = ['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']
df_join3[date_ts_cols] = df_join3[date_ts_cols].astype(str)

# delete certain columns 
df_join3 = df_join3.drop(columns=['SB_UUID'], axis=1) # because it messes up the export
# delete because exporting size causes problems
drop_cols = [
     # from roof_kat
     'WAERMEERTRAG', 'DUSCHGAENGE', 'DG_HEIZUNG', 'DG_WAERMEBEDARF', 'BEDARF_WARMWASSER', 
     'BEDARF_HEIZUNG', 'FLAECHE_KOLLEKTOREN', 'VOLUMEN_SPEICHER', 
     # from pv
     'MainCategory', 'PlantCategory', 
     # from gm
     'UUID', 'DATUM_AEND', 'DATUM_ERST', 'ERSTELL_J', 'ERSTELL_M', 'REVISION_J', 'REVISION_M',
     'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M', 'OBJEKTART', 'REVISION_Q', 
     'ICC', 'GEM_TEIl', 'GEM_FLAECH', 'SHN']

df_export = df_join3.drop(drop_cols, axis=1)
print(f'*dropped unnecessary columns | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
df_join3.info()

df_join3.to_file(f'{data_path}/temp_cache/df_joined_solkat_pv_municipality.shp')
print(f'*exported df_join3 to shp | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
df_join3.to_file(f'{data_path}/temp_cache/df_joined_solkat_pv_municipality.geojson', driver='GeoJSON')  # GeoJSON format
print(f'*exported df_join3 to geojson | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')
df_join3.to_csv(f'{data_path}/temp_cache/df_joined_solkat_pv_municipality.csv', index = False)
print(f'*exported df_join3 to csv | time: {datetime.now()} | runtime: {datetime.now() - start_timer} h:mm:ss.ss')

print(f'\n\n\n ************\nscript finished\n************\n\n\n')
winsound.Beep(440, 100) # frequency, duration


# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.75 GiB for an array with shape (41, 15543557) and data type float64

# RuntimeError: GDAL Error: Failed to write shape object. The maximum file size of 4294967188 has been reached. The current record of size 428 cannot be added.. Failed to write record: <fiona.model.Feature object at 0x000002DED4A6F410>
# (optimalpv-rh-py3.11) PS D:\RaulHochuli_inuse\OptimalPV_RH>
