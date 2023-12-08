import numpy as np
import geopandas as gpd
import functions

# ----------------------------------------------------------------------------------------------------------------
# Data gathering for Lupien
# ----------------------------------------------------------------------------------------------------------------
roof_kat = gpd.read_file(f'{data_path}/input/roof_kataster_2020-01-01/roof_kataster_2020-01-01.gpkg')
set_buffer = 1.25
roof_agg = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
checkpoint_to_logfile(f'end GROUPBY', n_tabs = 2)
roof_agg.to_file(f'{data_path}/temp_cache/roof_agg_res.shp')


# ----------------------------------------------------------------------------------------------------------------
# Code until here in TESTING with server
# ----------------------------------------------------------------------------------------------------------------
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
pv1 = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
bldng_reg = gpd.read_file(f'{data_path}/temp_cache/bldng_reg_1110_1121_1122_1130.shp')
kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')

pv1.set_crs(kt_shp.crs, allow_override=True, inplace=True )
bldng_reg.set_crs(kt_shp.crs, allow_override=True, inplace=True)

buffered_geom = pv1.buffer(1, resolution = 16)
pv2 = pv1.copy()
pv2['geometry'] = buffered_geom
pv2.set_crs(kt_shp.crs, allow_override=True, inplace=True)

pv2.to_file(f'{data_path}/temp_cache/pv2_buff1.shp')

bldng_reg.shape
pv2.shape
elec_prod.shape

pv_to_bldng = gpd.sjoin(pv2, bldng_reg, how = "left", predicate = "intersects")
pv_to_bldng.info()
grouped = pv_to_bldng.groupby('buildingCl')['TotalPower']


pv_to_bldng['buildingCl'].value_counts()
"""    
    1110 Gebäude mit einer Wohnung
      - Einzelhäuser wie Bungalows, Villen, Chalets, Forsthäuser, Bauernhäuser, Landhäuser usw.
      - Doppel- und Reihenhäuser, wobei jede Wohnung ein eigenes Dach und einen eigenen ebenerdigen Eingang hat
    1121 Gebäude mit zwei Wohnungen
      - Einzel-, Doppel- oder Reihenhäuser mit zwei Wohnungen
    1122 Gebäude mit drei oder mehr Wohnungen
      - Sonstige Wohngebäude wie Wohnblocks mit drei oder mehr Wohnungen
    1130 Wohngebäude für Gemeinschaften
      - Wohngebäude, in denen bestimmte Personen gemeinschaftlich wohnen, einschliesslich der Wohnungen für ältere Menschen, Studenten, Kinder
        und andere soziale Gruppen, z.B. Altersheime, Heime für Arbeiter, Bruderschaften, Waisen, Obdachlose usw.
"""
single = sum(pv_to_bldng['buildingCl']==1110)
double = sum(pv_to_bldng['buildingCl']==1121)
tripmor = sum(pv_to_bldng['buildingCl']==1122)
multiple =sum(pv_to_bldng['buildingCl']==1130)

tot_pv = pv2.shape[0]
(-(single + double + tripmor + multiple) - tot_pv) / tot_pv
single/tot_pv
double/tot_pv
tripmor/tot_pv
multiple/tot_pv

single =  pv_to_bldng.loc[sum(pv_to_bldng['buildingCl']==1110)]
double =  pv_to_bldng.loc[sum(pv_to_bldng['buildingCl']==1121)]
tripmo = pv_to_bldng.loc[sum(pv_to_bldng['buildingCl']==1122)]
multip = pv_to_bldng.loc[sum(pv_to_bldng['buildingCl']==1130)]

single['TotalPower'].mean()
double['TotalPower'].mean()
tripmo['TotalPower'].mean()
multip['TotalPower'].mean()
# TODO: hist_plot over all groups :) then that should show how single houses are justified to be sub selected for Monte Carlo iteration

roof_agg_withBldngReg = gpd.sjoin(roof_agg, bldng_reg, how="left", op="intersects")


# ----------------------------------------------------------------------------------------------------------------
# Code until here in TESTING with server
# ----------------------------------------------------------------------------------------------------------------


# associate aggregated roofs to building regisnter ----------------------------------------------------------------
kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
roof_kat = gpd.read_file(f'{data_path}/temp_cache/roof_kat_1_2_4_8_19_20.shp', rows=10000)
bldng_reg = gpd.read_file(f'{data_path}/temp_cache/bldng_reg_1110_1121_1122_1130.shp', rows=10000)
roof_agg = gpd.read_file(f'{data_path}/temp_cache/roof_agg_res.shp', rows=10000)
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows=10000)
pv1 = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()



# add pv to agg roofs
# roof_agg_withPV = gpd.sjoin(roof_agg, pv, how="left", op="within")
# roof_agg_withPV.to_file(f'{data_path}/temp_cache/roof_agg_withPV.shp')     

# roof_agg_withPV.info()
# roof_agg_withPV['SB_UUID']
# roof_agg_withPV['index_right']

# add buliding reg to agg roofs 
roof_agg_withBldngReg = gpd.sjoin(roof_agg, bldng_reg, how="left", op="intersects")
# roof_agg_withBldngReg.to_file(f'{data_path}/temp_cache/roof_agg_withBldngReg.shp')

roof_agg_withBldngReg.shape[0] / 1000000
bldng_reg.shape[0] / 1000000
roof_agg.shape[0] / 1000000

roof_agg_withBldngReg.info()
roof_agg_withBldngReg['SB_UUID'].value_counts()
roof_agg_withBldngReg['egid'].value_counts()
roof_agg_withBldngReg['buildingCl'].value_counts()


roof_agg_withBldngReg['egid'].isna().sum()
roof_agg_withBldngReg.index
bldng_reg.index

# Check for 'index_left' and 'index_right' columns and rename them if they exist
if 'index_left' in roof_agg_withBldngReg.columns:
    roof_agg_withBldngReg.rename(columns={'index_left': 'index_left_old'}, inplace=True)
if 'index_right' in roof_agg_withBldngReg.columns:
    roof_agg_withBldngReg.rename(columns={'index_right': 'index_right_old'}, inplace=True)


roof_kat.set_crs(kt_shp.crs, allow_override=True, inplace=True)  
bldng_reg.set_crs(kt_shp.crs, allow_override=True, inplace=True)
roof_agg.set_crs(kt_shp.crs, allow_override=True, inplace=True)
pv1.set_crs(kt_shp.crs, allow_override=True, inplace=True)
nan_egid = roof_agg_withBldngReg[roof_agg_withBldngReg['egid'].isna()]
nearest = gpd.sjoin_nearest(nan_egid, bldng_reg, how='left', distance_col='distances')





# ----------------------------------------------------------------------------------------------------------------




idx = roof_agg.index[102]
i = 0
a=0
b = 0
c = 0
for idx, row_srs in roof_agg.iterrows():
     bldng_IN_roof = bldng_reg.within(roof_agg.loc[idx, 'geometry'])
     if sum(bldng_IN_roof) == 1:
          roof_agg.loc[idx, 'bldng_reg_egid'] = bldng_reg.loc[bldng_IN_roof, 'egid'].values[0]
          a = a+1
     elif sum(bldng_IN_roof) > 1: 
          roof_agg.loc[idx, 'bldng_reg_egid'] = str(bldng_reg.loc[bldng_IN_roof, 'egid'].values)          
          b = b+1
     elif sum(bldng_reg.within(roof_agg.loc[idx, 'geometry'])) == 0:
          roof_agg.loc[idx, 'bldng_reg_egid'] = 0
          c = c+1
     i=i+1
     print(f'roof_agg loop:  {i} of {len(roof_agg)} done,      => a={a}, b={b}, c={c}')





# ----------------------------------------------------------------------------------------------------------------
# BOOKMARK 
# ----------------------------------------------------------------------------------------------------------------

# create single shape per roof_ID ---------------------------------------------------------------------------------
#TODO: aggregate to roof_union
#TODO: match roof shapes to building register
#TODO: extend match of "non-assigned" to building register
#TODO: match roof_kataster to building
#TODO: match facade kataster to building
#TODO: match pv to building

# ----------------------------------------------------------------------------------------------------------------
# END 
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('END of main_file.py')
if script_run_on_server == 0:
     winsound.Beep(400, 100)
     winsound.Beep(400, 100)
     winsound.Beep(400, 500)





     