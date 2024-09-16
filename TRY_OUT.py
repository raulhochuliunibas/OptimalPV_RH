import os
import pandas as pd
import numpy as np
import json
import winsound
import glob

scen = "pvalloc_smallBL_10y_npv_weighted"
wd_path = 'C:/Models/OptimalPV_RH'
data_path = f'{wd_path}_data'

# ------------------------------------------------------------------------------------------------------
"""

topo = json.load(open(f'{data_path}/output/{scen}/topo_egid.json', 'r'))

# topo characteristics
topo[list(topo.keys())[0]].keys()
topo[list(topo.keys())[0]]['pv_inst']


all_topo_m = glob.glob(f'{data_path}/output/pvalloc_smallBL_SLCTN_npv_weighted/interim_predictions/topo*.json')

for f in all_topo_m: 

    topo = json.load(open(f, 'r'))
    print(f'\ncounts for {f.split("topo_")[-1].split(".json")[0]}')

    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    for k,v in topo.items():
        # print(k)
        egid_list.append(k)
        gklas_list.append(v['gwr_info']['gklas'])
        inst_tf_list.append(v['pv_inst']['inst_TF'])
        inst_info_list.append(v['pv_inst']['info_source'])
        if 'xtf_id' in v['pv_inst']:
            inst_id_list.append(v['pv_inst']['xtf_id'])
        else:   
            inst_id_list.append('')
        beginop_list.append(v['pv_inst']['BeginOp'])
        power_list.append(v['pv_inst']['TotalPower'])
    # for ls in [egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list]:
    #     print(len(ls))

    topo_df = pd.DataFrame({'egid': egid_list, 'gklas': gklas_list, 'inst_tf': inst_tf_list, 'inst_info': inst_info_list, 
                            'inst_id': inst_id_list, 'beginop': beginop_list, 'power': power_list})
    
"""

if False:
    topo = json.load(open(f'{data_path}/output\pvalloc_smallBL_1y_npv_weighted/topo_egid.json', 'r'))
    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    for k,v in topo.items():
        # print(k)
        egid_list.append(k)
        gklas_list.append(v['gwr_info']['gklas'])
        inst_tf_list.append(v['pv_inst']['inst_TF'])
        inst_info_list.append(v['pv_inst']['info_source'])
        if 'xtf_id' in v['pv_inst']:
            inst_id_list.append(v['pv_inst']['xtf_id'])
        else:   
            inst_id_list.append('')
        beginop_list.append(v['pv_inst']['BeginOp'])
        power_list.append(v['pv_inst']['TotalPower'])

    topo_df = pd.DataFrame({'egid': egid_list, 'gklas': gklas_list, 'inst_tf': inst_tf_list, 'inst_info': inst_info_list,
                            'inst_id': inst_id_list, 'beginop': beginop_list, 'power': power_list})

    # topo_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_df.parquet')

    topo_df.to_csv(f'{wd_path}/topo3_df.csv')
# ------------------------------------------------------------------------------------------------------


import statistics as stats
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def rand_skew_norm(fAlpha, fLocation, fScale):
    sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

    afRN = np.random.randn(2)
    u0 = afRN[0]
    v = afRN[1]
    u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

    if u0 >= 0:
        return u1*fScale + fLocation 
    return (-u1)*fScale + fLocation 

def randn_skew(N, skew=0.0):
    return [rand_skew_norm(skew, 0, 1) for x in range(N)]

n_sample = 10**5
df_before = pd.DataFrame({'skew3': randn_skew(n_sample, 3), 'skew0': randn_skew(n_sample, 0)})
df_before['stand'] = (df_before['skew3'] / df_before['skew3'].max())
df_before['id'] = df_before.index

df = df_before.copy()
df_pick_list = []
draws = 1000
print('\n\nstart loop')
for i in range(1, draws+1):
    if (i) % (draws/4)==0:
        print(f'{i/ draws * 100}% done')
    rand_num = np.random.uniform(0, 1)
    df['stand'] = (df['skew3'] / df['skew3'].max())
    df['diff_stand_rand'] = abs(df['stand'] - rand_num)
    df_pick  = df[df['diff_stand_rand'] == min(df['diff_stand_rand'])].copy()

    if df_pick.shape[0] > 1:
        print('more than one row picked')
        rand_row = np.random.randint(0, df_pick.shape[0])
        df_pick = df_pick.iloc[rand_row]
    # adjust df
    df_pick_list.append(df_pick)
    df = df.drop(df_pick.index)
df_picked = pd.concat(df_pick_list)

# hist_data = [df_before['skew0'],df['skew0'], df_before['stand'], df['stand'],]
# labels = ['skew0_before', 'skew0_after', 'stand_before', 'stand_after']

hist_data = [df_before['skew3'],df['skew3'], df_picked['skew3'], df_before['stand'], df['stand'], ]
labels = ['skew3_before', 'skew3_after', 'skew3_picked', 'stand_before', 'stand_after']
df_picked['skew3'].var()
df_picked['stand'].var()

df_before.shape, df.shape, df_picked.shape
print('create fig')
fig = ff.create_distplot(hist_data, labels, bin_size=0.005)
fig.show()
print('end loop')



winsound.Beep(500, 1000)
winsound.Beep(500, 1000)


