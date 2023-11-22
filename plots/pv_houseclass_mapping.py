import os as os
import functions
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
import winsound
import plotly 
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import glob
import scipy
import plotly.figure_factory as ff


from functions import chapter_to_logfile, checkpoint_to_logfile

from plotly.subplots import make_subplots
from datetime import datetime


# pre run settings -----------------------------------------------------------------------------------------------
script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server

# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------

# pre setup + working directory ----------------------------------------------------------------------------------

# set working directory
timer = datetime.now()
if script_run_on_server == 0:
    winsound.Beep(840,  100)
    wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
    wd_path = "D:\OptimalPV_RH"         # path for server directory
data_path = f'{wd_path}_data'
os.chdir(wd_path)
with open(f'pv_proposal_pvShare_to_bldng.txt', 'w') as log_file:
    log_file.write(f' \n')

# import plot data -----------------------------------------------------------------------------------------------------
# probably covered by code further down
"""
if True:
    # all electricity production plants
    elec_prod_path = f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg'

    # import data frames
    elec_prod = gpd.read_file(f'{elec_prod_path}/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
    elec_prod.info()
    elec_prod['SubCategory'].value_counts()

    # change data types
    elec_prod['BeginningOfOperation'] = pd.to_datetime(elec_prod['BeginningOfOperation'], format='%Y-%m-%d')

    # aggregate data -----------------------------------------------------------------------------------------------------

    # change date col name
    cols = elec_prod.columns.tolist()
    cols[5] = 'Date'
    elec_prod.columns = cols

    # changing category names
    elec_prod['SubCategory'].value_counts()
    case = [
    (elec_prod['SubCategory'] == "subcat_1"),
    (elec_prod['SubCategory'] == "subcat_2"),
    (elec_prod['SubCategory'] == "subcat_3"),
    (elec_prod['SubCategory'] == "subcat_4"),
    (elec_prod['SubCategory'] == "subcat_5"),
    (elec_prod['SubCategory'] == "subcat_6"),
    (elec_prod['SubCategory'] == "subcat_7"),
    (elec_prod['SubCategory'] == "subcat_8"),
    (elec_prod['SubCategory'] == "subcat_9"),
    (elec_prod['SubCategory'] == "subcat_10"),]
    when = ['Hydro', 'PV', 'Wind', 'Biomass', 'Geothermal', 'Nuclear', 'Oil', 'Gas', 'Coal', 'Waste']

    elec_prod['SubCategory'] = np.select(case, when)


    # Grouping by month and 'SubCategor', then aggregating by count and sum of 'TotalPow'
    select_col = 'Counts'  # 'Counts' or 'PowerSum'
    elec_prod_ts_agg = elec_prod.groupby([elec_prod['Date'].dt.to_period("Y"), 'SubCategory']).agg(
        Counts=('xtf_id', 'size'), 
        PowerSum=('InitialPower', 'sum')
        ).reset_index()

    elec_prod_ts_agg['Date'] = elec_prod_ts_agg['Date'].dt.to_timestamp() # Convert 'BeginningO' back to timestamp for plotting
    elec_prod_ts_agg.info()
"""

# House charactersitics TO PV installations TABLE EXPORT -----------------------------------------------------------
use_raw_data_to_compute = True
buffer_size_list = np.arange(0, 7, 0.2)
absolute_value_cutoff = 10000  # for example

if True:    
    chapter_to_logfile('PV installations over House types.py', log_file_name='pv_proposal_pvShare_to_bldng.txt')
    
    def agg_pv_to_bldng_by_group(pv_to_bldng_func):
        pv_to_bldng_func['buildingCl'] = pv_to_bldng_func['buildingCl'].fillna('NA')
        pv_to_bldng_func_group = pv_to_bldng_func.groupby(['buildingCl']).agg(
            counts = ('TotalPower', 'size'),
            share = ('TotalPower', 'size'), 
            capacity = ('TotalPower', 'sum'), 
            capacity_avg =('TotalPower', 'mean'), 
            ).reset_index()
        pv_to_bldng_func_group['share'] = pv_to_bldng_func_group['share'] / pv_to_bldng_func_group['share'].sum()
        return pv_to_bldng_func_group
        
    if use_raw_data_to_compute:

        old_files = glob.glob(f'{data_path}/temp_cache/pv_to_bldng_*')
        for file in old_files:
            if os.path.exists(file):
                os.remove(file)

        elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        pv0 = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
        pv0.to_file(f'{data_path}/temp_cache/elec_prod_ONLY_pv.shp')
        bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')   #gpd.read_file(f'{data_path}/temp_cache/bldng_reg_1110_1121_1122_1130.shp')
        kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')

        pv0.set_crs(kt_shp.crs, allow_override=True, inplace=True )
        bldng_reg.set_crs(kt_shp.crs, allow_override=True, inplace=True)
        
        for buffer_size in buffer_size_list:
            buffered_geom = pv0.buffer(buffer_size, resolution = 16)
            pv = pv0.copy()
            pv['geometry'] = buffered_geom
            pv.set_crs(kt_shp.crs, allow_override=True, inplace=True)
            pv_to_bldng = gpd.sjoin(pv, bldng_reg, how = "left", predicate = "intersects")
            pv_to_bldng['geometry'] = pv_to_bldng['geometry'].centroid
            pv_to_bldng.to_file(f'{data_path}/temp_cache/pv_to_bldng_{buffer_size}.shp')
            print(f'buffer_size: {buffer_size} done')

        pv_to_bldng_GLOB = []
        pv_to_bldng_group_GLOB = []
        names_pv_to_bldng_GLOB = glob.glob(f'{data_path}/temp_cache/pv_to_bldng_*.shp')

    elif not use_raw_data_to_compute:
        # bldng_reg = gpd.read_file(f'{data_path}/temp_cache/bldng_reg_1110_1121_1122_1130.shp')
        # pv_to_bldng = gpd.read_file(f'{data_path}/temp_cache/pv_to_bldng.shp')

        pv_to_bldng_GLOB = []
        pv_to_bldng_group_GLOB = []
        names_pv_to_bldng_GLOB = glob.glob(f'{data_path}/temp_cache/pv_to_bldng_*.shp')

    for name_df in names_pv_to_bldng_GLOB:
        df = gpd.read_file(name_df)
        df.rename(columns={'buildingClass': 'buildingCl'}, inplace=True) if "bulidingCl" in df.columns else df
        df_group = agg_pv_to_bldng_by_group(df)

        pv_to_bldng_GLOB.append(df)
        pv_to_bldng_group_GLOB.append(df_group)
        print(name_df)

    for group in pv_to_bldng_group_GLOB:
        with open(f'pv_proposal_pvShare_to_bldng.txt', 'a') as log_file:
            log_file.write(f' \n\n')
            checkpoint_to_logfile(f'group: {group}', log_file_name='pv_proposal_pvShare_to_bldng.txt')
            log_file.write(group.to_string(index=False))

chapter_to_logfile('finished matching PV to House types for buffers 0.025 to 10', 
                     log_file_name='pv_proposal_pvShare_to_bldng.txt')
winsound.Beep(840,  100)
winsound.Beep(840,  100)
winsound.Beep(840,  200)
winsound.Beep(840,  100)

# ----------------------------------------------------------------------------------------------------------------
#  Bookmark - Distribution plot
# ----------------------------------------------------------------------------------------------------------------




# assign NA values a label
# Define the specified buildingCl categories
pv_to_bldng_1 = gpd.read_file(f'{data_path}/temp_cache/pv_to_bldng_1.shp')
specified_labels = ['1110', '1121', '1122', '1130']

pv_to_bldng_1 = pv_to_bldng_1.dropna(subset=['buildingCl'])
if pv_to_bldng_1['buildingCl'].dtype == 'float64':
    pv_to_bldng_1['buildingCl'] = pv_to_bldng_1['buildingCl'].astype(int)
    pv_to_bldng_1['buildingCl'] = pv_to_bldng_1['buildingCl'].astype(str)

pv_to_bldng_1['buildingCl'].dtype
pv_to_bldng_1.info()
pv_to_bldng_1['buildingCl'][0:10]

# Initialize a dictionary to collect TotalPower values for each category
total_power_data = {str(label): [] for label in specified_labels}
total_power_data['rest'] = []  # For all other categories

absolute_value_cutoff = 2000
# Populate the dictionary with the TotalPower values
for index, row in pv_to_bldng_1.iterrows():
    building_class = row['buildingCl']
    total_power = row['TotalPower']
    if total_power <= absolute_value_cutoff:
        if pd.isna(building_class):
            total_power_data['NA'].append(total_power)
        elif building_class in specified_labels:
            total_power_data[str(building_class)].append(total_power)
        else:
            total_power_data['rest'].append(total_power)

data = [total_power for total_power in total_power_data.values() if total_power]
labels = [label for label, total_power in total_power_data.items() if total_power]

# Now create the distribution plot
fig = ff.create_distplot(data, labels, show_rug=True)

# Show the figure
fig.show()


# ----------------------------------------------------------------------------------------------------------------

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


#-------------------------------------------------------------------------------------------------------------------

pv_to_bldng['buildingCl'].value_counts()
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
