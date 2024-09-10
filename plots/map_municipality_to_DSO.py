import os as os
import pandas as pd

import geopandas as gpd

import winsound
# import functions
# import numpy as np

# import matplotlib.pyplot as plt
# import pyogrio

# import plotly 
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.offline as pyo
# import glob
# import scipy
# import plotly.figure_factory as ff


# from functions import chapter_to_logfile, checkpoint_to_logfile

# from plotly.subplots import make_subplots
# from datetime import datetime


# -------------------------------
# Setup + Import
# -------------------------------

wd_path = 'C:/Models/OptimalPV_RH'
data_path = f'{wd_path}_data'

os.chdir(wd_path)
os.listdir(f'{data_path}')

           
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGRENZE.shp')
elcom = pd.read_csv(f'{data_path}/input/elcom-data-2024.csv', sep=',')

elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
elec_prod.columns
elec_prod['SubCategory'].unique()
pv_df = elec_prod[(elec_prod['SubCategory'] == 'subcat_2') & 
                  (elec_prod['TotalPower'] < 100)].copy()

pv_df.info()
pv_df['BeginningOfOperation'] = pd.to_datetime(pv_df['BeginningOfOperation'])
pv_df['BeginningOfOperation'].min()
pv_df['BeginningOfOperation'].max()
winsound.Beep(440, 250)

