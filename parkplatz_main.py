
# subset bldng_reg to relevant building classes ------------------------------------------------------------------
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
buildingClass_residential = [1110, 1121, 1122, 1130]
bldng_reg_residential = bldng_reg.loc[bldng_reg['buildingClass'].isin(buildingClass_residential)].copy()    
bldng_reg_residential.to_file(f'{data_path}/z_py_exports/bldng_reg_residential.shp')
checkpoint_to_logfile(f'subset bldng_reg to relevant building classes')


# prep new df from roof_kat, unionize all partitions per building  -----------------------------------------------

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
roof_kat_sub = roof_kat.loc[roof_kat['SB_UUID'].isin(roof_kat['SB_UUID'].unique()[0:2000])].copy()

sb_obj_unique = roof_kat_sub['SB_UUID'].unique() 
roof_union = gpd.GeoDataFrame(index = sb_obj_unique, columns = ['geometry']) #TODO: find a better naming convetion!
roof_union.set_crs(roof_kat.crs, inplace=True)


# create new df from roof_kat: loop over each building merging shapes-------------------------------------------------
idx = '{AF378B63-B28F-4A92-9BEB-4B84ABD75BDF}' #TODO: delete later when no longer used
set_buffer = 1.25 # determines the buffer around shapes to ensure a more proper union merge of single ouse shapes
for idx, row_srs in roof_union.iterrows():
    
    # add unified geometry
    #row_srs['geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union
    #roof_union.loc[idx, 'geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union
    roof_union.loc[idx, 'geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16) # roof_geom_buff.buffer(-0.5, resolution = 16).copy()

checkpoint_to_logfile(f'unionized roof parts per building')
roof_union.to_file(f'{data_path}/roof_union_1_W{set_buffer}buffer.shp')


# intersect roof shape with buildingClass, roof_union with bldng_reg_residential ---------------------------------
# TODO: NEXT STEPS:
# TODO: 1. create a loop, singling out each building again, similar to the loop above
# TODO: 2. intersect the single roof shape with the bldng_reg_residential, add all relevant information to roof_union
# TODO: 3. export the file into proper format
# TODO: 4. create proper reading part that imports the intermediary results later so that I don't always have to run the whole script up to here. 

bldng_reg_residential.columns

roof_union['n_match_bldng_reg'] = np.nan
roof_union['bldngStatus'] = np.nan
roof_union['bldngCategory'] = np.nan 
roof_union['bldngClass'] = np.nan

idx = '{AF378B63-B28F-4A92-9BEB-4B84ABD75BDF}' #TODO: delete later when no longer used
for idx, row_srs in roof_union.iterrows():
    
    bldng_IN_idx = bldng_reg_residential['geometry'].intersects(roof_union.loc[idx, 'geometry'])
    roof_union.loc[idx, 'n_match_bldng_reg'] = bldng_IN_idx.sum()
    if bldng_IN_idx.sum() == 1:
        roof_union.loc[idx, 'bldngStatus'] = bldng_reg_residential.loc[bldng_IN_idx, 'buildingStatus'].values[0]
        roof_union.loc[idx, 'bldngCategory'] = bldng_reg_residential.loc[bldng_IN_idx, 'buildingCategory'].values[0]
        roof_union.loc[idx, 'bldngClass'] = bldng_reg_residential.loc[bldng_IN_idx, 'buildingClass'].values[0]

checkpoint_to_logfile(f'extended roof_union with bldng_reg_residential info')
roof_union.to_file(f'{data_path}/roof_union_2_AFTER_bldng_extension.shp')

roof_union.columns
roof_union['n_match_bldng_reg'].value_counts()  



# export ---------------------------------------------------------------------------------------------------------
roof_union.to_file(f'{data_path}/z_py_exports/roof_union.shp')
roof_union.to_file(f'{data_path}/z_py_exports/roof_union.geojson', driver='GeoJSON')
checkpoint_to_logfile(f'aggregated gdf exported')




# ----------------------------------------------------------------------------------------------------------------
# END 
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('END of main_file.py')
if script_run_on_server == 0:
     winsound.Beep(400, 100)
     winsound.Beep(400, 100)
     winsound.Beep(400, 500)




# TO-DO LIST: ----------------------------------------------------------------------------------------------------
# TODO: intersect building use with unnion DF, then ,kick out all non residential buildings




# ----------------------------------------------------------------------------------------------------------------
# CODE FOR ADDIGN LATER!!! 
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# Add other data to roof based dataframe
# ----------------------------------------------------------------------------------------------------------------

cols = ['FLAECHE', 'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG']
cats = ['cat2_', 'cat3_', 'cat4_', 'cat5_']
new_col = [cat + col for cat in cats for col in cols ]
roof_union[new_col] = np.nan
checkpoint_to_logfile(f'created empty df to then iter over roof parts')


# create new df from roof_kat: loop over each building merging shapes---------------------------------------------
idx = '{AF378B63-B28F-4A92-9BEB-4B84ABD75BDF}'
set_buffer = 1 # determines the buffer around shapes to ensure a more proper union merge of single ouse shapes
for idx, row_srs in roof_union.iterrows():
    
    # add unified geometry
    #row_srs['geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union
    #roof_union.loc[idx, 'geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].unary_union
    roof_union.loc[idx, 'geometry'] = roof_kat_sub.loc[roof_kat_sub['SB_UUID'] == idx, 'geometry'].buffer(set_buffer, resolution = 16).unary_union.buffer(-set_buffer, resolution = 16) # roof_geom_buff.buffer(-0.5, resolution = 16).copy()

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
roof_union.to_file(f'{data_path}/roof_union_W{set_buffer}buffer_PILOT_bldgINTERSECTION.shp')





# define roof range which are considered
cutoff_roof_kat_area = [10,300] #TODO: add here values from the PV installation data set

# pv inst: add empty nan conlumns to aggr df   -------------------------------------------------------------------

roof_union['pv_d', 'pv_address', 'pv_postcode', 'pv_BeginningO', 'InitialPow', 'TotalPow' ] = np.nan


# pv inst: loop over pv   ----------------------------------------------------------------------------------------


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
# ----------------------------------------------------------------------------------------------------------------
# end - Add other data to roof based dataframe
# ----------------------------------------------------------------------------------------------------------------
