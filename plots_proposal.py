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
from plotly.subplots import make_subplots

from functions import chapter_to_logfile, checkpoint_to_logfile
from datetime import datetime
from shapely.ops import unary_union

import warnings


# pre run settings -----------------------------------------------------------------------------------------------
script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server


# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------


# pre setup + working directory ----------------------------------------------------------------------------------

# create log file for checkpoint comments
timer = datetime.now()
with open(f'plots_proposal_log.txt', 'w') as log_file:
        log_file.write(f' \n')
chapter_to_logfile('started plots_proposal.py')

# set working directory
if script_run_on_server == 0:
     winsound.Beep(840,  100)
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)

# import data -----------------------------------------------------------------------------------------------------

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
elec_prod_ts_agg = elec_prod.groupby([elec_prod['Date'].dt.to_period("W"), 'SubCategory']).agg(
     Counts=('xtf_id', 'size'), 
     PowerSum=('InitialPower', 'sum')
     ).reset_index()

elec_prod_ts_agg['Date'] = elec_prod_ts_agg['Date'].dt.to_timestamp() # Convert 'BeginningO' back to timestamp for plotting
elec_prod_ts_agg.info()

# plot ELECTRICITS PROD OVER TIME --------------------------

def plot_pv_counts_ts(show_plot = True):

    # plotting
    fig = px.line(elec_prod_ts_agg, x='Date', y=f'{select_col}', color='SubCategory', title=f'Monthly {select_col} of SubCategor Entries')
    fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    # zoom plot
    start_date = '2000-01-01'
    end_date = '2023-12-31'
    fig.update_xaxes(range=[start_date, end_date])


    # add time steps
    cap_date1 = "2009-01-01"
    fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=cap_date1,
        x1=cap_date1,
        y0=0.025,
        y1=0.975,
        yref="paper",
        line=dict(color="black", width=1)
    )
    )
    fig.add_annotation(
    x=cap_date1,
    y=0.95,
    yref="paper",
    text="KEV, 2009",
    showarrow=False,
    font=dict(color="black")
    )

    cap_date2 = "2014-01-01"
    fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=cap_date2,
        x1=cap_date2,
        y0=0.025,
        y1=0.975,
        yref="paper",
        line=dict(color="black", width=1)
    )
    )
    fig.add_annotation(
    x=cap_date2,
    y=0.95,
    yref="paper",
    text="EIV, 2014",
    showarrow=False,
    font=dict(color="black")
    )
    
    if show_plot: 
        fig.show()
plot_pv_counts_ts(False)

# plot PV Counts + PowerSum OVER TIME -----------------------------------
def plot_pv_2axis_counts_capacity_ts(show_plot = True):

    # Filter the dataset for SubCategory 'PV'
    df_pv = elec_prod_ts_agg[elec_prod_ts_agg['SubCategory'] == 'PV']

    # Create a subplot with secondary y-axis
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for Counts + PowerSum
    fig2.add_trace(
        go.Scatter(x=df_pv['Date'], y=df_pv['Counts'], mode='lines', name='Counts'),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(x=df_pv['Date'], y=df_pv['PowerSum'], mode='lines', name='PowerSum'),
        secondary_y=True,
    )

    # Update the layout
    fig2.update_layout(
        title_text='Monthly Counts and PowerSum of SubCategory PV Entries',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    fig2.update_yaxes(title_text="Counts", showgrid=True, gridcolor='lightgray', secondary_y=False)
    fig2.update_yaxes(title_text="PowerSum", showgrid=True, gridcolor='lightgray', secondary_y=True)

    # zoom plot
    start_date = '2003-01-01'
    end_date = '2023-12-31'
    fig2.update_xaxes(range=[start_date, end_date])

    # add time steps
    cap_date1 = "2009-01-01"
    fig2.add_shape(
        go.layout.Shape(
            type="line",
            x0=cap_date1,
            x1=cap_date1,
            y0=0.025,
            y1=0.975,
            yref="paper",
            line=dict(color="black", width=1)
        )
    )
    fig2.add_annotation(
        x=cap_date1,
        y=0.95,
        yref="paper",
        text="KEV, 2009",
        showarrow=False,
        font=dict(color="black")
    )

    cap_date2 = "2014-01-01"
    fig2.add_shape(
        go.layout.Shape(
            type="line",
            x0=cap_date2,
            x1=cap_date2,
            y0=0.025,
            y1=0.975,
            yref="paper",
            line=dict(color="black", width=1)
        )
    )
    fig2.add_annotation(
        x=cap_date2,
        y=0.95,
        yref="paper",
        text="EIV, 2014",
        showarrow=False,
        font=dict(color="black")
    )
    if show_plot:
        fig2.show()

plot_pv_2axis_counts_capacity_ts()



elec_prod['Date']