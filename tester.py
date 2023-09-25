import os as os
import pandas as pd
import numpy as np

wd_path = f'C:\\Models\\OptimalPV_RH'
wd_path_data = f'{wd_path}_data'   

os.listdir(f'{wd_path_data}\\SHEDS')
sheds17 = pd.read_csv(f'{wd_path_data}\\SHEDS\\SHEDS2017.csv', sep = ',', )
sheds18 = pd.read_csv(f'{wd_path_data}\\SHEDS\\SHEDS2018.csv', sep = ',', )
sheds19 = pd.read_csv(f'{wd_path_data}\\SHEDS\\SHEDS2019.csv', sep = ',', )
sheds20 = pd.read_csv(f'{wd_path_data}\\SHEDS\\SHEDS2020.csv', sep = ',', )
sheds21 = pd.read_csv(f'{wd_path_data}\\SHEDS\\SHEDS2021.csv', sep = ',', )

all_id2 = pd.concat([sheds17['id'], sheds18['id'], sheds19['id'], sheds20['id'], sheds21['id']], axis = 0)

all_id = all_id2.drop_duplicates()
all_id.shape[0] / all_id2.shape[0]  

df_aggr = pd.DataFrame({'id':all_id})
    
id = df_aggr['id'][0]

df_aggr.shape[0]
df_aggr.index

for index, row in df_aggr.iterrows():
    print(f'index: {index} of {df_aggr.shape[0]}')


df_aggr.loc[(df_aggr['id'] == id), 'in_sheds17'] = sheds17.loc[(sheds17['id'] == id)]['accom8_1'] == 'yes'  
df_aggr.loc[(df_aggr['id'] == id), 'in_sheds18'] = sheds18.loc[(sheds18['id'] == id)]['accom8_1'] == 'yes'
df_aggr.loc[(df_aggr['id'] == id), 'in_sheds19'] = sheds19.loc[(sheds19['id'] == id)]['accom8_1'] == 'yes'
df_aggr.loc[(df_aggr['id'] == id), 'in_sheds20'] = sheds20.loc[(sheds20['id'] == id)]['accom8_1'] == 'yes'
df_aggr.loc[(df_aggr['id'] == id), 'in_sheds21'] = sheds21.loc[(sheds21['id'] == id)]['accom8_1'] == 'yes'




df_aggr.to_csv(f'{wd_path_data}\\SHEDS\\SHEDS_id_aggregated_over_years.csv', sep = ',', index = False)

    

    
# bookmark --------------------------------------------------------------------------------

# def id_in_shedsdf(df_id, df_sheds):
#     #print(df_id['id'])
#     in_set = (df_sheds['id'] == (df_id['id'])) & (df_sheds['accom8_1'] == 'yes' )
#     in_set.sum()
#     return in_set

# df_aggr['in_sheds17'] = df_aggr.apply(lambda x: id_in_shedsdf(x, sheds17), axis = 1)

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

import geopandas as gpd
import pyogrio
import winsound

winsound.Beep(840,  100)

wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
data_path = f'{wd_path}_data'

kt_shp = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
pv = gpd.read_file(f'{data_path}/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
pv.set_crs(kt_shp.crs, allow_override = True, inplace = True)
pv.crs == kt_shp.crs
winsound.Beep(840,  100)
winsound.Beep(840,  100)


sub_kt_number = 15
kt_shp.loc[kt_shp['KANTONSNUM'] == sub_kt_number, ['NAME', 'KANTONSNUM']] 
kt_shp_sub = kt_shp.loc[kt_shp["KANTONSNUM"] == sub_kt_number,].copy()


# pv_sub = gpd.sjoin(pv, kt_shp_sub, how='left')
# type(pv['geometry'])
# type(kt_shp_sub['geometry'])
# pv_sub = pv['geometry'].intersection(kt_shp_sub['geometry'])
# pv['geometry'].intersects(kt_shp_sub).value_counts()
# pv['geometry'].within(kt_shp_sub).value_counts()
# pv['geometry'].contains(kt_shp_sub).value_counts()
pv_sub = gpd.overlay(pv, kt_shp_sub, how='intersection')
pv_sub.head()
type(pv_sub)

winsound.Beep(840,  100)

kt_shp_sub.to_file(f'{data_path}/subsample_faster_run/kt_shp_sub.shp')	
pv_sub.to_file(f'{data_path}/subsample_faster_run/pv_sub6.shp')
asdf = r'C:\Models\OptimalPV_RH_data\subsample_faster_run\pv_sub7.shp'
pv_sub.to_file(asdf)
# pv_sub.to_file(f'{data_path}/subsample_faster_run/pv_sub3.gpkg', layer = 'pv_sub3', driver = 'GPKG')

pv_sub.to_file( f'C:\Models\OptimalPV_RH_data\subsample_faster_run\pv_sub.gpkg', layer = 'pv_sub3', driver = 'GPKG')
winsound.Beep(840,  100)
winsound.Beep(840,  100)

