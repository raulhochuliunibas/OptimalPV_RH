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

from functions import chapter_to_logfile, checkpoint_to_logfile

from plotly.subplots import make_subplots
from datetime import datetime


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
elec_prod_ts_agg = elec_prod.groupby([elec_prod['Date'].dt.to_period("Y"), 'SubCategory']).agg(
     Counts=('xtf_id', 'size'), 
     PowerSum=('InitialPower', 'sum')
     ).reset_index()

elec_prod_ts_agg['Date'] = elec_prod_ts_agg['Date'].dt.to_timestamp() # Convert 'BeginningO' back to timestamp for plotting
elec_prod_ts_agg.info()


# plot ELECTRICITS PROD OVER TIME ------------------------------------------------------------------------------------
def plot_pv_counts_ts(show_plot = True, export_plot = False):

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

# plot PV Counts + PowerSum OVER TIME  -------------------------------------------------------------------------------
title_font = 1
legend_title_font = 20 # 40
leged_font = 15 #35
tick_title_font = 20 #40
tick_font = 15# 35
caption_font = 20 #40
line_size = 3 # 7
start_date = '1993-01-01'
end_date = '2023-12-31'

def add_date_caption(fig_func, date_func, str_func, y0_func=0.05, y1_func=0.925):
    cap_date = date_func
    fig_func.add_shape(
        go.layout.Shape(
            type="line",
            x0=cap_date,
            x1=cap_date,
            y0=y0_func,
            y1=y1_func,
            yref="paper",
            line=dict(color="black", width=1, dash='dash')
        )
    )
    fig_func.add_annotation(
        x=cap_date,
        y=0.95,
        yref="paper",
        text=str_func,
        showarrow=False,
        font=dict(color="black", size = caption_font)
    )
def plot_pv_2axis_counts_capacity_ts(show_plot = True, export_plot = False):

    # Filter the dataset for SubCategory 'PV'
    df_pv = elec_prod_ts_agg[elec_prod_ts_agg['SubCategory'] == 'PV']

    # Create a subplot with secondary y-axis
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for counts + counPowreSum
    fig2.add_trace(
        go.Scatter(x=df_pv['Date'], y=df_pv['Counts'], mode='lines', name='Counts', line=dict(color = 'blue', width=line_size)),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(x=df_pv['Date'], y=df_pv['PowerSum'], mode='lines', name='PowerSum', line=dict(color = 'red', width=line_size)),
        secondary_y=True,
    )

    # Update the layout
    fig2.update_layout(
        title_text='Anuall Counts and Capacity of PV Plants in Switzerland',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    fig2.update_yaxes(title_text="Counts", showgrid=True, gridcolor='lightgray', secondary_y=False)
    fig2.update_yaxes(title_text="PowerSum", showgrid=True, gridcolor='lightgray', secondary_y=True)

    # zoom plot
    # start_date = '2002-01-01'
    # end_date = '2023-12-31'
    fig2.update_xaxes(range=[start_date, end_date])

    add_date_caption(fig2, "2009-01-01", "KEV, 2009", 0.025, 0.925)
    add_date_caption(fig2, "2014-01-01", "EIV, 2014", )

    # update axes
    fig2.update_xaxes(
        title_font=dict(size=tick_title_font),  # Font size for X-axis title
        tickfont=dict(size=tick_font)     # Font size for X-axis tick labels
    )
    fig2.update_yaxes(
        title_font=dict(size=tick_title_font),  # Font size for Y-axis title
        tickfont=dict(size=tick_font)     # Font size for Y-axis tick labels
    )
    fig2.update_layout(
        title_font=dict(size=title_font),    # Font size for figure title
        legend_title_font=dict(size=legend_title_font),  # Font size for legend title
        legend_font=dict(size=leged_font)    # Font size for legend items
    )

    # change trace name   
    fig2.update_traces(name="sum Capacity (kW)", selector=dict(name="PowerSum"))
    
    fig2.update_layout(showlegend = False)

    if show_plot:
        fig2.show()
    if export_plot:
        pyo.plot(fig2, filename=f'{data_path}/output/plot_pv_2axis_counts_capacity_ts.html', auto_open=False)
    
    return fig2

plot_pv_2axis_counts_capacity_ts(show_plot=True, export_plot=False)

# plot PV Growth OVER TIME -----------------------------------------------------------------------------------------
def plot_pv_2axis_counts_capacity_growth(show_plot = True, export_plot = False):
    elec_prod_ts_agg = elec_prod.groupby([elec_prod['Date'].dt.to_period("Y"), 'SubCategory']).agg(
        Counts=('xtf_id', 'size'), 
        PowerSum=('InitialPower', 'sum')
        ).reset_index()
    elec_prod_ts_agg['Date'] = elec_prod_ts_agg['Date'].dt.to_timestamp() # Convert 'BeginningO' back to timestamp for plotting
    elec_prod_ts_agg.info()

    df_pv = elec_prod_ts_agg[elec_prod_ts_agg['SubCategory'] == 'PV'].copy()
    # calcucalte growth rates
    df_pv['Counts_growth'] = (df_pv['Counts'] - df_pv['Counts'].shift(1)) / df_pv['Counts'].shift(1)
    df_pv['PowerSum_growth'] = (df_pv['PowerSum'] - df_pv['PowerSum'].shift(1)) / df_pv['PowerSum'].shift(1)

    # fig_func = make_subplots(specs=[[{"secondary_y": True}]])
    fig_func = go.Figure()
    df_pv.head(10)

    # add traces
    fig_func.add_trace(go.Scatter(x=df_pv['Date'], y=df_pv['Counts_growth'], mode='lines', name='Counts Growth', line=dict(color = 'blue', width=line_size)))
    fig_func.add_trace(go.Scatter(x=df_pv['Date'], y=df_pv['PowerSum_growth'], mode='lines', name='PowerSum Growth', line=dict(color = 'red', width=line_size)))

    # Update the layout
    fig_func.update_layout(
        title_text='Anuall Growth of Counts and Capacity of PV Plants in Switzerland',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),)

    # zoom plot
    # start_date = '1997-01-01'
    # end_date = '2023-12-31'
    fig_func.update_xaxes(range=[start_date, end_date])

    add_date_caption(fig_func, "2009-01-01", "KEV, 2009")
    add_date_caption(fig_func, "2014-01-01", "EIV, 2014")

    # update axes
    fig_func.update_xaxes(
        title_font=dict(size=tick_title_font),  # Font size for X-axis title
        tickfont=dict(size=tick_font))
    fig_func.update_yaxes(
        title_font=dict(size=tick_title_font),  # Font size for Y-axis title
        tickfont=dict(size=tick_font))
    fig_func.update_layout(
        title_font=dict(size=title_font),    # Font size for figure title
        legend_title_font=dict(size=legend_title_font),  # Font size for legend title
        legend_font=dict(size=leged_font))    # Font size for legend items

    fig_func.update_traces(name="growth Capacity (kW)", selector=dict(name="PowerSum"))
    fig_func.update_layout(showlegend = False)

    if show_plot:
        fig_func.show()
    if export_plot:
        pyo.plot(fig_func, filename=f'{data_path}/output/plot_pv_2axis_counts_capacity_growth.html', auto_open=False)

plot_pv_2axis_counts_capacity_growth(show_plot=True, export_plot=False)



