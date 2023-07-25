import pandas as pd
import numpy as np
import os as os
import geopandas as gpd
import xarray as xr
import datashader as ds 
import contextily as ctx
import matplotlib as mpl
import pyogrio

from shapely import geometry

wd_path = "C:/Models/OptimalPV"
data_path = "C:/Models/data"
os.chdir(wd_path)   
os.listdir()
os.listdir(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb')

ch_kt = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
wgs84_crs = ch_kt.crs.to_string().split(" +up")[0]
ch_kt = ch_kt.to_crs(wgs84_crs)

ch_gm = gpd.read_file(f'{data_path}/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
wgs84_crs = ch_gm.crs.to_string().split(" +up")[0]
ch_gm = ch_gm.to_crs(wgs84_crs)

#dat_solkat = gpd.read_file(f'{data_path}/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
"""
ch_kt = ch_kt.to_crs(new_crs)



dat_solkat <- st_read(dsn = paste0(wd_path,"/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb"),layer = "SOLKAT_CH_DACH")

"""


print("****************************** \nfinished running main_file.py \n******************************")
