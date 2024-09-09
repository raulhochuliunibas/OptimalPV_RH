import os
import pandas as pd
import numpy as np
import json
import winsound
import glob


wd_path = 'C:/Models/OptimalPV_RH'
data_path = f'{wd_path}_data'

topo = json.load(open(f'{data_path}/output/pvalloc_smallBL_SLCTN_npv_weighted/topo_egid.json', 'r'))

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
    



topo = json.load(open(f'{data_path}/output/pvalloc_smallBL_SLCTN_npv_weighted/topo_egid.json', 'r'))
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

topo_df.to_csv(f'{wd_path}/topo_df.csv')
print(topo_df['PV_INST'].value_counts())