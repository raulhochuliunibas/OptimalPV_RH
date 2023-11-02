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


# pre run settings -----------------------------------------------------------------------------------------------
script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server
subsample_faster_run = 0          # 0 = run on all data, 
                                  # 1 = run on SWEET municipalities subset for faster run
                                  # 2 = run on residential data points
create_data_subsample = 1         # 0 = do not create data subsample, 1 = create data subsample


# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------


# pre setup + working directory ----------------------------------------------------------------------------------

# create log file for checkpoint comments
timer = datetime.now()
with open(f'main_file_log.txt', 'w') as log_file:
        log_file.write(f' \n')
chapter_to_logfile('started running main_file.py')

# set working directory
if script_run_on_server == 0:
     winsound.Beep(840,  100)
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)


# import geo referenced data -------------------------------------------------------------------------------------

# load administrative shapes
kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
#kt_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
checkpoint_to_logfile(f'finished loading administrative shp', n_tabs = 2)

if subsample_faster_run == 0:
    # load solar kataster shapes
    roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
    #roof_kat = gpd.read_file(f'{data_path}/temp_cache/roof_kat_1_2_4_8_19_20.shp')
    checkpoint_to_logfile(f'finished loading roof solar kataster shp', n_tabs = 1)
    #faca_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-fassaden_2056.gdb/SOLKAT_FASS_20230221.gdb', layer ='SOLKAT_CH_FASS')
    faca_kat = roof_kat.copy()
    checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)

    # load building register indicating residential or industrial use
    bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
    checkpoint_to_logfile(f'finished loading building register pt', n_tabs = 2)

    # load heating / cooling demand raster 150x150m
    heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
    checkpoint_to_logfile(f'finished loading heat & cool demand pt', n_tabs = 1)

    # load pv installation points
    # pv = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
    elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
    pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
    checkpoint_to_logfile(f'finished loading pv installation pt', n_tabs = 2) 
    
    # check if all CRS are compatible
    kt_shp.crs == gm_shp.crs == roof_kat.crs == faca_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs
    gm_shp.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    roof_kat.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    faca_kat.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    bldng_reg.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    heatcool_dem.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    pv.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    
    all_crs_equal = kt_shp.crs == gm_shp.crs == roof_kat.crs == faca_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs
    if all_crs_equal:
         checkpoint_to_logfile(f'CRS are compatible', n_tabs = 1)
    elif not all_crs_equal:
     checkpoint_to_logfile(f'CRS are NOT compatible', n_tabs = 1)
     raise Exception('CRS are NOT compatible')

    
    # export ONLY residential data points to temp cache ---------------------------------------------------------------
    # this should help in the beginning for preliminary analyses
    
    # convert some date/time to str for export to shp file
    # roof_kat subset and export 
    checkpoint_to_logfile(f'\tstart subset for roof_kat ONLY Residential', n_tabs = 2)
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
    roof_kat.columns
    roof_kat.info()
    roof_kat[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']] = roof_kat[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']].astype(str)
    
    cat_sb_object = [1,2,4,8,19,20]
    roof_kat_res = roof_kat.loc[roof_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
    roof_kat_res['SB_OBJEKTART'].value_counts()
    roof_kat_res.to_file(f'{data_path}/temp_cache/residential_subsample/roof_kat_1_2_4_8_19_20.shp')
    checkpoint_to_logfile(f'\t\t * finished subset roof_kat_1_2_4_8_19_20', n_tabs = 1)

    # faca_kat subset and export
    faca_kat.columns
    faca_kat.info()
    faca_kat[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']] = faca_kat[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']].astype(str)

    cat_sb_object = [1,2,4,8] #TODO: check if this is the same as for roof_kat
    faca_kat_res = faca_kat.loc[faca_kat['SB_OBJEKTART'].isin(cat_sb_object)].copy()
    faca_kat_res.to_file(f'{data_path}/temp_cache/residential_subsample/faca_kat_1_2_4_8.shp')
    checkpoint_to_logfile(f'\t\t * finished subset faca_kat_1_2_4_8', n_tabs = 1)
    
    # bldng_reg: subset to relevant bulidings in bldng_reg 
    """
    See here for building codes that are selected:
    https://www.bfs.admin.ch/bfs/de/home/register/gebaeude-wohnungsregister/inhalt-referenzdokumente.assetdetail.22905270.html
    
    "buildingStatus":
    1001 Projektiert
    1002 Bewilligt
    1003 Im Bau
    1004 Bestehend
    1005 Nicht nutzbar
    1007 Abgebrochen
    1008 Nicht realisiert
    
    "builingCategory":
    0    ??
    1010 Provisorische Unterkunft
    1020 Gebäude mit ausschliesslicher Wohnnutzung
    1030 Andere Wohngebäude (Wohngebäude mit Nebennutzung)
    1040 Gebäude mit teilweiser Wohnnutzung
    1060 Gebäude ohne Wohnnutzung
    1080 Sonderbau
    
    "buildingClass":
    0    ??
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
        
    1211 Hotelgebäude
    1212 Andere Gebäude für kurzfristige Beherbergungen
    1220 Bürogebäude
    1230 Gross- und Einzelhandelsgebäude
    1231 Restaurants und Bars in Gebäuden ohne Wohnnutzung
    1241 Bahnhöfe, Abfertigungsgebäude, Fernsprechvermittlungszentralen
    1242 Garagengebäude
    1251 Industriegebäude
    1252 Behälter, Silos und Lagergebäude
    1261 Gebäude für Kultur- und Freizeitzwecke
    1262 Museen / Bibliotheken
    1263 Schul- und Hochschulgebäude, Forschungseinrichtungen
    1264 Krankenhäuser und Facheinrichtungen des Gesundheitswesens
    1265 Sporthallen
    1271 Landwirtschaftliche Betriebsgebäude
    1272 Kirchen und sonstige Kulturgebäude
    1273 Denkmäler oder unter Denkmalschutz stehende Bauwerke
    1274 Sonstige Hochbauten, anderweitig nicht genannt
    1275 Andere Gebäude für die kollektive Unterkunft
    1276 Gebäude für die Tierhaltung
    1277 Gebäude für Pflanzenbau
    1278 Andere landwirtschaftliche Betriebsgebäude
    """
    bldng_reg.columns
    bldng_reg.info()
    bldng_reg.buildingClass

    buildingClass_res = [1110, 1121, 1122, 1130]
    bldng_reg_res = bldng_reg.loc[bldng_reg['buildingClass'].isin(buildingClass_res)].copy()    
    bldng_reg_res.to_file(f'{data_path}/temp_cache/residential_subsample/bldng_reg_1110_1121_1122_1130.shp')
    checkpoint_to_logfile(f'\t\t * finished subset bldng_reg_1110_1121_1122_1130', n_tabs = 1)


    # export subsamples SWEET municipalities for faster run ----------------------------------------------------------
    if create_data_subsample == 1:
          kt_number_sub = [16,]
          gm_number_sub = [3851, 3901, 4761, 1083, 4001, 1061, 2829, 4042 ]
          kt_shp.loc[kt_shp['KANTONSNUM'].isin(kt_number_sub), ['NAME', 'KANTONSNUM']] 
          gm_shp.loc[gm_shp['BFS_NUMMER'].isin(gm_number_sub), ['NAME', 'BFS_NUMMER']]

          # create folder for subsample shapes selected, remove old files if they exist
          if not os.path.exists(f'{data_path}/temp_cache/sweet_subsample'):
               os.makedirs(f'{data_path}/temp_cache/sweet_subsample')
          elif os.path.exists(f'{data_path}/temp_cache/sweet_subsample'):
               for file in os.listdir(f'{data_path}/temp_cache/sweet_subsample'):
                    os.remove(f'{data_path}/temp_cache/sweet_subsample/{file}')

          # create subsample shapes selected
          checkpoint_to_logfile(f'\tstart creating subsamples', n_tabs = 3)
          kt_shp_sub = kt_shp.loc[kt_shp["KANTONSNUM"].isin(kt_number_sub),].copy()
          gm_shp_sub = gm_shp.loc[gm_shp['BFS_NUMMER'].isin(gm_number_sub),].copy()
          checkpoint_to_logfile(f'\t * finished subsetting admin shp', n_tabs = 2)
          
          roof_kat_sub = gpd.sjoin(roof_kat, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting roof_kat', n_tabs = 3)
          
          faca_kat_sub =      gpd.sjoin(faca_kat, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting faca_kat', n_tabs = 3)
          bldng_reg_sub =     gpd.sjoin(bldng_reg, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting bldng_reg', n_tabs = 2)
          heatcool_dem_sub =  gpd.sjoin(heatcool_dem, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting heatcool_dem', n_tabs = 2)
          pv_sub =             gpd.sjoin(pv, gm_shp_sub, how="inner", op="within")   
          checkpoint_to_logfile(f'\t * finished subsetting pv', n_tabs = 3)
          
          # convert some date/time to str for export to shp file
          roof_kat_sub[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']] = roof_kat[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']].astype(str)
          faca_kat_sub[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']] = faca_kat[['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG']].astype(str)
          pv.info()


          # export subsample to shape files
          checkpoint_to_logfile(f'\tstart exporting subsample shapes', n_tabs = 2)
          gm_shp_sub.to_file(f'{data_path}/temp_cache/sweet_subsample/gm_shp_sub.shp')
          roof_kat_sub.to_file(f'{data_path}/temp_cache/sweet_subsample/roof_kat_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting roof_kat_sub', n_tabs = 3)
          faca_kat_sub.to_file(f'{data_path}/temp_cache/sweet_subsample/faca_kat_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting faca_kat_sub', n_tabs = 3)
          bldng_reg_sub.to_file(f'{data_path}/temp_cache/sweet_subsample/bldng_reg_sub.shp')
          heatcool_dem_sub.to_file(f'{data_path}/temp_cache/sweet_subsample/heatcool_dem_sub.shp')
          pv_sub.to_file(f'{data_path}/temp_cache/sweet_subsample/pv_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting subsample to shape files', n_tabs = 1)

     
elif subsample_faster_run == 1:
     checkpoint_to_logfile(f'using SUBSAMPLE(1) for faster run', n_tabs = 1)

     #load subset shapes
     os.listdir(f'{data_path}/temp_cache/sweet_subsample')   
     roof_kat = gpd.read_file(f'{data_path}/temp_cache/sweet_subsample/roof_kat_sub.shp')
     checkpoint_to_logfile(f'finished loading roof solar kataster shp', n_tabs = 1)
     #faca_kat = gpd.read_file(f'{data_path}/temp_cache/sweet_subsample/faca_kat_sub.shp')
     checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)
     bldng_reg = gpd.read_file(f'{data_path}/temp_cache/sweet_subsample/bldng_reg_sub.shp')
     checkpoint_to_logfile(f'finished loading building register pt', n_tabs = 2)
     heatcool_dem = gpd.read_file(f'{data_path}/temp_cache/sweet_subsample/heatcool_dem_sub.shp')
     checkpoint_to_logfile(f'finished loading heat & cool demand pt', n_tabs = 1)
     pv = gpd.read_file(f'{data_path}/temp_cache/sweet_subsample/pv_sub.shp')
     checkpoint_to_logfile(f'finished loading pv installation pt', n_tabs = 2)


elif subsample_faster_run == 2:
     checkpoint_to_logfile(f'using ONLY RESIDENTIAL(2) data subsample', n_tabs = 1)

     #load subset shapes
     os.listdir(f'{data_path}/temp_cache')
     roof_kat = gpd.read_file(f'{data_path}/temp_cache/roof_kat_1_2_4_8_19_20.shp')
     checkpoint_to_logfile(f'finished loading roof solar kataster shp', n_tabs = 1)
     # faca_kat = gdp.read_file(f'{data_path}/temp_cache/faca_kat_1_2_4_8.shp')
     checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)
     bldng_reg = gpd.read_file(f'{data_path}/temp_cache/bldng_reg_residential.shp')
     checkpoint_to_logfile(f'finished loading building register pt', n_tabs = 2)
     
     heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
     checkpoint_to_logfile(f'finished loading heat & cool demand pt', n_tabs = 1)
     pv = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
     checkpoint_to_logfile(f'finished loading pv installation pt', n_tabs = 2)

     
# import regular, nonGIS data --------------------------------------------------------------------------------------------
# dict_elec_prod_dispatch = {'Week':['05.01.2022', '12.01.2022', '19.01.2022', '26.01.2022', '02.02.2022', '09.02.2022', '16.02.2022', '23.02.2022', '02.03.2022', '09.03.2022', '16.03.2022', '23.03.2022', '30.03.2022', '06.04.2022', '13.04.2022', '20.04.2022', '27.04.2022', '04.05.2022', '11.05.2022', '18.05.2022', '25.05.2022', '01.06.2022', '08.06.2022', '15.06.2022', '22.06.2022', '29.06.2022','06.07.2022', '13.07.2022', '20.07.2022', '27.07.2022', '03.08.2022', '10.08.2022', '17.08.2022', '24.08.2022', '31.08.2022', '07.09.2022', '14.09.2022', '21.09.2022', '28.09.2022', '05.10.2022', '12.10.2022', '19.10.2022', '26.10.2022', '02.11.2022', '09.11.2022', '16.11.2022', '23.11.2022', '30.11.2022', '07.12.2022', '14.12.2022', '21.12.2022', '28.12.2022'],
#                          'consumption_Gwh':[195.2, 236.1, 216.7, 214.7, 218.6, 205.4, 213.2, 198.0, 196.9, 207.3, 193.0, 194.1, 191.4, 191.6, 170.2, 159.3, 167.3, 173.2, 164.4, 150.7, 158.1, 163.3, 161.3, 161.4, 175.0, 159.3, 150.1, 158.7, 144.4, 152.8, 149.6, 156.8, 144.2, 158.8, 171.6, 162.0, 172.0, 164.6, 180.7, 162.7, 173.5, 168.9, 171.1, 173.3, 190.3, 182.4, 199.0, 204.6, 220.4, 201.1, 208.0, 171.7]} 
# elec_dem_2022 = pd.DataFrame(dict_elec_prod_dispatch)
#TODO: import electricity demand data properly
checkpoint_to_logfile(f'finished loading electricity demand 2022(non-standardized for other years)', n_tabs = 1)



# ----------------------------------------------------------------------------------------------------------------
# Aggregate roof_kat for house shapes  
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('Create House Based Dataframe')
if script_run_on_server == 0: 
     winsound.Beep(840,  100)
     winsound.Beep(840,  100)

roof_kat = gpd.read_file(f'{data_path}/temp_cache/roof_kat_1_2_4_8_19_20.shp')
checkpoint_to_logfile(f'\n\nstart GROUPBY unionizing ', n_tabs = 1)
# create new df and aggregate roof parts
set_buffer = 1.25
roof_agg = roof_kat.groupby('SB_UUID')['geometry'].apply(lambda x: x.buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16))
checkpoint_to_logfile(f'end GROUPBY', n_tabs = 2)
roof_agg.to_file(f'{data_path}/temp_cache/roof_agg_res.shp')

# ----------------------------------------------------------------------------------------------------------------
# BOOKMARK runs ok until here
# ----------------------------------------------------------------------------------------------------------------




# ----------------------------------------------------------------------------------------------------------------
# F*** it and just try the never ending loop :(
# ----------------------------------------------------------------------------------------------------------------
sb_uuid = roof_kat['SB_UUID'].unique()
roof_agg = gpd.GeoDataFrame(index = sb_uuid, columns = ['geometry'])
roof_agg.set_crs(roof_kat.crs, allow_override=True, inplace=True)

idx = '{AF378B63-B28F-4A92-9BEB-4B84ABD75BDF}' #TODO: delete later when no longer used
set_buffer = 1.25 # determines the buffer around shapes to ensure a more proper union merge of single ouse shapes
i=0
for idx, row_srs in roof_agg.iterrows():
    # add unified geometry
    roof_agg.loc[idx, 'geometry'] = roof_kat.loc[roof_kat['SB_UUID'] == idx, 'geometry'].buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16) # roof_geom_buff.buffer(-0.5, resolution = 16).copy()
    i=i+1
    #print(f'roof_agg loop, {i} of {len(roof_agg)} done')



winsound.Beep(840,  100)


roof_agg.to_file(f'{data_path}/temp/roof_agg_allCH.shp')







# ----------------------------------------------------------------------------------------------------------------



# associate aggregated roofs to building regisnter ----------------------------------------------------------------
roof_agg['bldng_reg_egid'] = np.nan


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


# prepare for PV_TOPO, house based dataframe --------------------------------------------------------------------- 




# create single shape per roof_ID ---------------------------------------------------------------------------------
#TODO: aggregate to roof_union
#TODO: match roof shapes to building register
#TODO: extend match of "non-assigned" to building register
#TODO: match roof_kataster to building
#TODO: match facade kataster to building
#TODO: match pv to building



# match roof shapes to building register



# ----------------------------------------------------------------------------------------------------------------
# END 
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('END of main_file.py')
if script_run_on_server == 0:
     winsound.Beep(400, 100)
     winsound.Beep(400, 100)
     winsound.Beep(400, 500)





     