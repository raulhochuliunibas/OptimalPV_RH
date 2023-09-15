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
