import os
import geopandas as gpd
import matplotlib.pyplot as plt

os.getcwd()

wd_path = 'C:/Models/OptimalPV_RH'
wd_data = f'{wd_path}_data'

os.listdir(wd_data)

df_gm = gpd.read_file(f'{wd_data}/swissboundaries3d_2023-01_2056_5728.shp', 
                      layer = 'swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET' )  
df_gm.columns

slct_gm_name = ['Davos', 'Chur', 'Sirnach', 'Buttisholz', 'Aarau', 'Luzern', 'Liestal', 'Turgi']
slct_gm_numb = [3851,     3901,    4761,     1083,         4001,    1061,     2829,       4042]
df_gm.loc[df_gm['NAME'].isin(slct_gm_name), ['NAME','BFS_NUMMER']] 

df_gm_sub = df_gm.loc[df_gm['NAME'].isin(slct_gm_name),].copy()
print(df_gm_sub[['NAME', 'BFS_NUMMER']])

df_gm_sub.to_file(f'{wd_data}/subsample_gm_OptimalPV_RH.shp')
df_gm_sub.head()

df_gm_sub.plot()
plt.show()
