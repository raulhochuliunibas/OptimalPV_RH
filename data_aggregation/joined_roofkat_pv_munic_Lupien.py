import os 
import pandas as pd
import numpy as np
import geopandas as gpd
import winsound

from shapely.ops import unary_union
from datetime import datetime


script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server

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
print(f'*imported gm_shp, time: {datetime.now() - start_timer} h:mm:ss.ss')
roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', 
                         layer ='SOLKAT_CH_DACH')
# roof_kat['SB_UUID'].nunique()
# roof_kat['SB_UUID'].head()
# roof_kat['GWR_EGID'].nunique()
# roof_kat['GWR_EGID'].head(50)

print(f'*imported roof_kat, time: {datetime.now() - start_timer} h:mm:ss.ss')
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
print(f'*imported elec_prod, time: {datetime.now() - start_timer} h:mm:ss.ss')
pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
print(f'*imported pv, time: {datetime.now() - start_timer} h:mm:ss.ss')   


# align CRSs
roof_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
pv.set_crs(gm_shp.crs, allow_override=True, inplace=True)
gm_shp.crs == roof_kat.crs == pv.crs

# # take only residential houses for now
# cat_sb_object = [1,]
# roof_kat = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
# print(f'*subset_roof_kat')

# unionize buffered polygons
set_buffer = 1.25
roof_agg_Srs = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
roof_agg = gpd.GeoDataFrame(roof_agg_Srs, geometry=roof_agg_Srs)
roof_agg.set_crs(gm_shp.crs, allow_override=True, inplace=True)
print(f'*unionized roof_kat')

# intersection of all 3 data sets --------------------------------------------------------------------------------
roof_kat.crs == gm_shp.crs == pv.crs == roof_agg.crs

df_join1 = gpd.sjoin(roof_agg, roof_kat, how = "left", predicate = "intersects")
df_join1.rename(columns={'index_right': 'index_roofkat'}, inplace=True)
df_join2 = gpd.sjoin(df_join1, pv, how = "left", predicate = "intersects")
df_join2.rename(columns={'index_right': 'index_pv'}, inplace=True)
df_join3 = gpd.sjoin(df_join2, gm_shp, how = "left", predicate = "intersects")
df_join3.rename(columns={'index_right': 'index_gm'}, inplace=True)


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
     'GEM_FLAECH']

df_export = df_join3.drop(drop_cols, axis=1)

df_join3.to_file(f'{data_path}/temp_cache/df_joined_solkat_pv_municipality.shp')
df_join3.to_csv(f'{data_path}/temp_cache/df_joined_solkat_pv_municipality.csv', index = False)

print(f'\n\n\n ************\nscript finished\n************\n\n\n')
winsound.Beep(440, 100) # frequency, duration


roof_kat
pv.info()


# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.75 GiB for an array with shape (41, 15543557) and data type float64

"""
Data columns (total 67 columns):
 #   Column                      Dtype
---  ------                      -----
 0   geometry                    geometry
 1   index_roofkat               float64
 2   DF_UID                      float64
 3   DF_NUMMER                   float64
 4   DATUM_ERSTELLUNG            datetime64[ns, UTC]
 5   DATUM_AENDERUNG             datetime64[ns, UTC]
 6   SB_UUID                     object
 7   SB_OBJEKTART                float64
 8   SB_DATUM_ERSTELLUNG         datetime64[ns, UTC]
 9   SB_DATUM_AENDERUNG          datetime64[ns, UTC]
 10  KLASSE                      float64
 11  FLAECHE                     float64
 12  AUSRICHTUNG                 float64
 13  NEIGUNG                     float64
 14  MSTRAHLUNG                  float64
 15  GSTRAHLUNG                  float64
 16  STROMERTRAG                 float64
 17  STROMERTRAG_SOMMERHALBJAHR  float64
 18  STROMERTRAG_WINTERHALBJAHR  float64
 19                  float64
 20  
 27  GWR_EGID                    float64
 28  SHAPE_Length                float64
 29  SHAPE_Area                  float64
 30  index_pv                    float64
 31  xtf_id                      float64
 32  Address                     object
 33  PostCode                    float64
 34  Municipality                object
 35  Canton                      object
 36  BeginningOfOperation        object
 37  InitialPower                float64
 38  TotalPower                  float64
 39  MainCategory                object
 40  SubCategory                 object
 41  PlantCategory               object
 42  index_gm                    float64
 43  UUID                        object
 44  DATUM_AEND                  object
 45  DATUM_ERST                  object
 46  ERSTELL_J                   float64
 47  ERSTELL_M                   object
 48  REVISION_J                  float64
 49  REVISION_M                  object
 50  GRUND_AEND                  object
 51  HERKUNFT                    object
 52  HERKUNFT_J                  float64
 53  HERKUNFT_M                  object
 54  OBJEKTART                   object
 55  BEZIRKSNUM                  float64
 56  SEE_FLAECH                  float64
 57  REVISION_Q                  object
 58  NAME                        object
 59  KANTONSNUM                  float64
 60  ICC                         object
 61  EINWOHNERZ                  float64
 62  HIST_NR                     float64
 63  BFS_NUMMER                  float64
 64  GEM_TEIL                    object
 65  GEM_FLAECH                  float64
 66  SHN                         object
dtypes: datetime64[ns, UTC](4), float64(40), geometry(1), object(22)
memory usage: 8.0+ GB"""