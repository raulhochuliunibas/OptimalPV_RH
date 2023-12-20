import os
import pandas as pd
import geopandas as gpd
import datetime
import glob

from datetime import datetime

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------

wd_path = 'C:/Models/OptimalPV_RH'
data_path = f'{wd_path}_data'
os.chdir(wd_path)
os.listdir(data_path)

# create a export txt file for summary outputs
print(f'\n\n ***** START SANITY CHECK ***** \t time: {datetime.now()}')
export_txt_name = f'{data_path}/sanitycheck_joined_roof_kat_pv_gm/sanity_check_output.txt'
with open(export_txt_name, 'w') as export_txt:
    export_txt.write(f'\n')
    export_txt.write(f'\n *************************** \n     SANITY CHECK OUTPUT \n *************************** \n')
    export_txt.write(f'\n* start script: time: {datetime.now()}')

# ------------------------------------------------------------------------------
# DATA IMPORT
# ------------------------------------------------------------------------------

# import data from input
subset = False
if not subset:
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
    elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
    pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
elif subset:
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET', rows = 10)
    roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH', rows = 10)
    elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 10)
    pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()


with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* imported all shapes from input: time: {datetime.now()}')

# import data from parquet
gm_shp_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/gm_shp.parquet')   
roof_kat_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/roof_kat_by_gm.parquet')
bldng_reg_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/bldng_reg_by_gm.parquet')
heatcool_dem_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/heatcool_dem_by_gm.parquet')
pv_pq = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/pv_by_gm.parquet')

with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* imported all shapes from parquet: time: {datetime.now()}')

# import aggregates 
agg_pq_files = glob.glob(f'{data_path}/agg_sol_kat_pv_BY_KT/agg_solkat_pv_gm_ALL/agg_solkat_pv_gm_*.parquet')
df_agg_pq = pd.DataFrame()
f = agg_pq_files[0]
for f in agg_pq_files:
    print(f)
    df_read = pd.read_parquet(f)
    df_agg_pq = df_agg_pq._append(pd.read_parquet(f))

with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* imported all aggregates from parquet: time: {datetime.now()}')

# ------------------------------------------------------------------------------
# START SANITY CHECKS
# ------------------------------------------------------------------------------

# compare number of roofs
roof_kat.columns
roof_kat_pq.columns
df_agg_pq.columns
with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* ------------------------------------------------------ \n* sanity check number of obs: time: {datetime.now()}')
    export_txt.write(f'\n* SB_UUID \t\t gm_shp \t\t gm_shp_pq \t\t df_agg_pq \n')
    export_txt.write(f'\n len()  \t\t\t {len(roof_kat["SB_UUID"])} \t\t {len(roof_kat_pq["SB_UUID"])} ')
    export_txt.write(f'\n nunique()   \t\t\t {len(roof_kat["SB_UUID"].unique())} \t\t {len(roof_kat_pq["SB_UUID"].unique())} ')
    export_txt.write(f'\n\n DF_UID \t\t gm_shp \t\t gm_shp_pq \t\t df_agg_pq \n')
    export_txt.write(f'\n len()  \t\t\t {len(roof_kat["DF_UID"])} \t\t {len(roof_kat_pq["DF_UID"])} \t\t {len(df_agg_pq["DF_UID"])}')
    export_txt.write(f'\n nunique()   \t\t\t {len(roof_kat["DF_UID"].unique())} \t\t {len(roof_kat_pq["DF_UID"].unique())} \t\t {len(df_agg_pq["DF_UID"].unique())}')



# compare production potential by roof kat
with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* ------------------------------------------------------ \n* sanity check production potential by roof kat: time: {datetime.now()}')
    export_txt.write(f'\n*  \t\t\t roof_kat \t\t roof_kat_pq \t\t df_agg_pq \n')
    export_txt.write(f'\n KLASSE 2')
    export_txt.write(f'\n len()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 2])} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 2])} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 2])} ')
    export_txt.write(f'\n nunique()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 2, "SB_UUID"].unique())} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 2, "SB_UUID"].unique())} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 2, "SB_UUID"].unique())} ')
    export_txt.write(f'\n sum() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 2, "STROMERTRAG"].sum(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 2, "STROMERTRAG"].sum(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 2, "STROMERTRAG"].sum(), 2)} ')
    export_txt.write(f'\n mean() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 2, "STROMERTRAG"].mean(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 2, "STROMERTRAG"].mean(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 2, "STROMERTRAG"].mean(), 2)} ')
    export_txt.write(f'\n KLASSE 3')
    export_txt.write(f'\n len()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 3])} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 3])} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 3])} ')
    export_txt.write(f'\n nunique()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 3, "SB_UUID"].unique())} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 3, "SB_UUID"].unique())} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 3, "SB_UUID"].unique())} ')
    export_txt.write(f'\n sum() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 3, "STROMERTRAG"].sum(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 3, "STROMERTRAG"].sum(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 3, "STROMERTRAG"].sum(), 2)} ')
    export_txt.write(f'\n mean() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 3, "STROMERTRAG"].mean(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 3, "STROMERTRAG"].mean(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 3, "STROMERTRAG"].mean(), 2)} ')
    export_txt.write(f'\n KLASSE 4')
    export_txt.write(f'\n len()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 4])} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 4])} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 4])} ')
    export_txt.write(f'\n nunique()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 4, "SB_UUID"].unique())} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 4, "SB_UUID"].unique())} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 4, "SB_UUID"].unique())} ')
    export_txt.write(f'\n sum() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 4, "STROMERTRAG"].sum(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 4, "STROMERTRAG"].sum(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 4, "STROMERTRAG"].sum(), 2)} ')
    export_txt.write(f'\n mean() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 4, "STROMERTRAG"].mean(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 4, "STROMERTRAG"].mean(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 4, "STROMERTRAG"].mean(), 2)} ')
    export_txt.write(f'\n KLASSE 5')
    export_txt.write(f'\n len()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 5])} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 5])} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 5])} ')
    export_txt.write(f'\n nunique()   \t\t\t {len(roof_kat.loc[roof_kat["KLASSE"] == 5, "SB_UUID"].unique())} \t\t {len(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 5, "SB_UUID"].unique())} \t\t {len(df_agg_pq.loc[df_agg_pq["KLASSE"] == 5, "SB_UUID"].unique())} ')
    export_txt.write(f'\n sum() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 5, "STROMERTRAG"].sum(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 5, "STROMERTRAG"].sum(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 5, "STROMERTRAG"].sum(), 2)} ')
    export_txt.write(f'\n mean() stromertrag \t\t {round(roof_kat.loc[roof_kat["KLASSE"] == 5, "STROMERTRAG"].mean(), 2)} \t\t {round(roof_kat_pq.loc[roof_kat_pq["KLASSE"] == 5, "STROMERTRAG"].mean(), 2)} \t\t {round(df_agg_pq.loc[df_agg_pq["KLASSE"] == 5, "STROMERTRAG"].mean(), 2)} ')


# compare pv installatoins
with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* ------------------------------------------------------ \n* sanity check number of pv installations: time: {datetime.now()}')
    export_txt.write(f'\n*  \t\t\t pv \t\t pv_pq \t\t df_agg_pq \n')
    export_txt.write(f'\n len()  \t\t\t {len(pv)} \t\t {len(pv_pq)} \t\t {len(df_agg_pq)} ')
    export_txt.write(f'\n nunique()xtf_id   \t\t {len(pv["xtf_id"].unique())} \t\t {len(pv_pq["xtf_id"].unique())} \t\t {len(df_agg_pq["xtf_id"].unique())} ')
    export_txt.write(f'\n sum() totalpower   \t\t {pv["TotalPower"].sum()} \t\t {pv_pq["TotalPower"].sum()} \t\t {df_agg_pq["TotalPower"].sum()} ')
    export_txt.write(f'\n mean() totalpower   \t\t {pv["TotalPower"].mean()} \t\t {pv_pq["TotalPower"].mean()} \t\t {df_agg_pq["TotalPower"].mean()} ')
    
print(f'\n\n ***** END SANITY CHECK ***** \t time: {datetime.now()}')