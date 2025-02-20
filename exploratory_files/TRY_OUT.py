import os
import pandas as pd
import numpy as np
import json
import winsound
import glob
import geopandas as gpd

wd_path = 'C:/Models/OptimalPV_RH'
data_path     = f'{wd_path}_data'
data_path_def = f'{wd_path}_data'
scen = "pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind"
name_dir_import = 'preprep_BLSO_22to23_1and2homes'



# ------------------------------------------------------------------------------------------------------
if True:
    egid = '410298'

    topo_subdf_pq_list = glob.glob(f'{data_path}/output/{scen}/topo_time_subdf/*.parquet')
    df_list= []
    for f in topo_subdf_pq_list:
        subdf = pd.read_parquet(f)
        if egid in subdf['EGID'].unique():
            #topo_time_subdf
            subdf = subdf.loc[subdf['EGID'] == egid]
            df_list.append(subdf)

            f_name = f.split(f'{data_path}/output/{scen}/topo_time_subdf\\')[1].split('.parquet')[0]
            print(f'EGID: {f_name} -> export csv')

    df = pd.concat(df_list)
    df.to_csv(f'{data_path}/output/{scen}/topo_time_subdf/{f_name}.csv')

    # solkat_month
    solkat_month = pd.read_parquet(f'{data_path}/output/{scen}/solkat_month.parquet')
    df.columns
    dfuid_ls = df['df_uid'].unique()
    solkat_month.loc[solkat_month['DF_UID'].isin(dfuid_ls)].to_csv(f'{data_path}/output/{scen}/topo_time_subdf/solkat_month_{egid}.csv') 


# ------------------------------------------------------------------------------------------------------
# print directory scheme to txt file
if False:
    from pathlib import Path

    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '


    def tree(dir_path: Path, prefix: str='', exclude_list = None):
        """A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        if exclude_list is None:
            exclude_list = []

        contents = [p for p in dir_path.iterdir() if p.name not in exclude_list and not p.name.startswith('.')]
        # contents each get pointers that are ├── with a final └── :
        pointers = ['├── '] * (len(contents) - 1) + ['└── ']
        for pointer, path in zip(pointers, contents):
            yield prefix + pointer + path.name
            if path.is_dir():  # extend the prefix and recurse:
                extension = '│   ' if pointer == '├── ' else '    '
                # i.e. space because last, └── , above so no more |
                yield from tree(path, prefix=prefix+extension, exclude_list=exclude_list)

    # Print the directory tree excluding specified directories and those starting with "."
    txt_header = f'** Directory structure for OptimalPV_RH **\n date: {pd.Timestamp.now()}\n\n'

    with open(f'{wd_path}/OptimalPV_RH_directory_structure.txt', 'w') as f:
        f.write(txt_header)
        for line in tree(Path('C:/Models/OptimalPV_RH'), exclude_list=['archiv_no_longer_used']):
            print(line)
            f.write(line + '\n')





# ------------------------------------------------------------------------------------------------------
"""
solkat_preprep_wo_missingEGID_gdf = gpd.read_file(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes\solkat_gdf.geojson")
solkat_preprep_incl_missingEGID_gdf = gpd.read_file(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\solkat_gdf.geojson")
gwr = pd.read_parquet(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\gwr.parquet")
# solkat = pd.read_parquet(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\solkat.parquet")


cols_to_check = ['391293', '391294', '391295', '391296', '391297', ]
# cols_to_check = ['391291', '391290']
subdf_preprep_incl_missingEGID = solkat_preprep_incl_missingEGID_gdf[solkat_preprep_incl_missingEGID_gdf['EGID'].isin(cols_to_check)]
subdf_preprep_wo_missingEGID = solkat_preprep_wo_missingEGID_gdf[solkat_preprep_wo_missingEGID_gdf['EGID'].isin(cols_to_check)]

subdf_preprep_incl_missingEGID.loc[:, ['EGID', 'geometry', 'FLAECHE', 'DF_UID']]
subdf_preprep_wo_missingEGID.loc[:, ['EGID', 'geometry', 'FLAECHE', 'DF_UID']]

subdf_preprep_wo_missingEGID.buffer(-0.5, resolution=16)

isin_ls = ['245054165', ' 245054165', '245054165 ', ' 245054165 ']
gwr.loc[gwr['EGID'].isin(isin_ls)]['EGID'].unique()

solkat_preprep_incl_missingEGID_gdf.loc[solkat_preprep_incl_missingEGID_gdf['EGID'].isin(isin_ls)]['EGID'].unique()
"""

# ------------------------------------------------------------------------------------------------------
"""
gpd.list_layers(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg')
solkat_month = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='SOLKAT_CH_DACH_MONAT', rows = 100000)
month = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='MONAT', rows = 1000)

solkat_month.to_csv(f'{wd_path}/solkat_month.csv')
solkat_month.dtypes
type(solkat_month)
solkat_month.head(20)
month.columns
                             
"""
# ------------------------------------------------------------------------------------------------------
"""
topo = json.load(open(f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'r'))

topo[list(topo.keys())[0]].get('pv_inst').get('info_source')

egid_ls, alloc_algorithm_ls = [], []
for k,v in topo.items():
    egid_ls.append(k)
    alloc_algorithm_ls.append(v.get('pv_inst').get('info_source'))

df = pd.DataFrame({'EGID': egid_ls, 'info_source': alloc_algorithm_ls})
df['info_source'].value_counts()
egids = df.loc[df['info_source'] == 'alloc_algorithm', 'EGID'].to_list()
egids 

# -----------------------
subdf_t0['EGID'].isin(egids).sum()
subdf['EGID'].isin(egids).sum()

aggsubdf_combo['EGID'].isin(egids).sum()

npv_df['EGID'].isin(egids).sum()
"""

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
# Theoretical change in NPV distribution over multiple allocation algorithms
if False:
        
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

# ------------------------------------------------------------------------------------------------------






