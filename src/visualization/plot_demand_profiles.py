# python integrated packages
import sys
import os as os
import json

# external packages
import plotly.graph_objects as go
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotDemandProfiles(VisualizationCore):
    """Plot of the pvalloc_initalization + sanitycheck stage."""

    def plot_demand_profiles(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_demand_profiles', self.visual_sett.log_name)

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # import 
            topo                     = json.load(open(f'{self.visual_sett.data_path}/pvalloc/{scen}/sanity_check_byEGID/topo_egid.json', 'r'))
            swstore_arch_typ_factors = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/swstore_arch_typ_factors.parquet')                    
            demandtypes_ts           = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/demandtypes_ts.parquet')

            # compute topo_df
            rows = []

            for k,v in topo.items():

                partitions = v.get('solkat_partitions', {})
                gwr_info = v.get('gwr_info', {})
                pv_inst = v.get('pv_inst', {})

                for k_p, v_p in partitions.items():
                    row = {
                        'EGID': k,
                        'df_uid': k_p,
                        'bfs': gwr_info.get('bfs'),
                        'GKLAS': gwr_info.get('gklas'),
                        'GAREA': gwr_info.get('garea'),
                        'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                        'demand_arch_typ': v.get('demand_arch_typ'),
                        'demand_elec_dem_pGAREA': v.get('demand_elec_dem_pGAREA'),
                        'grid_node': v.get('grid_node'),

                        'inst_TF': pv_inst.get('inst_TF'),
                        'info_source': pv_inst.get('info_source'),
                        'pvid': pv_inst.get('xtf_id'),
                        'pvtarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
                        'TotalPower': pv_inst.get('TotalPower'),
                        'df_uid_w_inst': v.get('df_uid_w_inst'),

                        'FLAECHE': v_p.get('FLAECHE'),
                        'AUSRICHTUNG': v_p.get('AUSRICHTUNG'),
                        'STROMERTRAG': v_p.get('STROMERTRAG'),
                        'NEIGUNG': v_p.get('NEIGUNG'),
                        'MSTRAHLUNG': v_p.get('MSTRAHLUNG'),
                        'elecpri_Rp_kWh': v.get('elecpri_Rp_kWh'),
                    }
                    rows.append(row)

            topo_df = pl.DataFrame(rows)

            demandtypes_mean_df = topo_df.group_by(['demand_arch_typ']).agg([
                pl.col('GAREA').mean().alias('GAREA_mean'),
                pl.col('GAREA').median().alias('GAREA_median'),
            ])                    
            demandtypes_mean_df = demandtypes_mean_df.with_columns(
                pl.col('demand_arch_typ').str.extract(r'^(SFH|MFH)', 1).alias('sfhmfh_typ')
            )


            # extend by demandtypes_ts + factors
            swstore_arch_typ_factors = swstore_arch_typ_factors.rename({'arch_typ': 'demand_arch_typ',
                                                                        'elec_dem_ind_cecb': 'demand_elec_pGAREA'})
            demandtypes_mean_df = demandtypes_mean_df.join(swstore_arch_typ_factors, on='demand_arch_typ', how='left')

            demandtypes_unpivot = demandtypes_ts.unpivot(
                on = ['SFH', 'MFH', ],
                index=['t', 't_int'],  # col that stays unchanged
                value_name='demand_profile',  # name of the column that will hold the values
                variable_name='sfhmfh_typ'  # name of the column that will hold the original column names
            )   
            demandtypes_mean_df = demandtypes_mean_df.join(demandtypes_unpivot, on='sfhmfh_typ', how='left')

            demandtypes_mean_df = demandtypes_mean_df.with_columns([
                (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_mean") ).alias("demand_kW_mean"),  
                (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_median")).alias("demand_kW_median")
            ])


            # aggregate demand per month
            demandtypes_monthly = demandtypes_ts.with_columns([
                pl.col("time").str.strptime(pl.Datetime, format="%d.%m.%y %H:%M").alias("parsed_time"),
                pl.col("time").str.strptime(pl.Datetime, format="%d.%m.%y %H:%M").dt.truncate("1mo").alias("month")
            ])
            demandtypes_monthly = demandtypes_monthly.group_by("month").agg([
                pl.col("MFH").sum().alias("MFH"),
                pl.col("SFH").sum().alias("SFH"),
                pl.col("t_int").min().alias("t_int")  # use min or first
            ])
            demandtypes_monthly = demandtypes_monthly.sort(['t_int',], descending=False, )  

            demandtypes_monthly_unpivot = demandtypes_monthly.unpivot(
                on = ['SFH', 'MFH', ],
                index=['t_int'],  # col that stays unchanged
                value_name='demand_profile',  # name of the column that will hold the values
                variable_name='sfhmfh_typ'  # name of the column that will hold the original column names
            )   
            demandtypes_mean_monthly = demandtypes_mean_df[['demand_arch_typ', 'GAREA_mean', 'GAREA_median', 'sfhmfh_typ', 'demand_elec_pGAREA' ]].join(demandtypes_monthly_unpivot, on='sfhmfh_typ', how='left')
            demandtypes_mean_monthly = demandtypes_mean_monthly.with_columns([
                (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_mean") ).alias("demand_kW_mean"),  
                (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA_median")).alias("demand_kW_median")
            ])

            # plot
            fig = go.Figure()
            for i, arch_typ in enumerate(demandtypes_mean_df['demand_arch_typ'].unique()):
                plotdf = demandtypes_mean_df.filter(pl.col('demand_arch_typ') == arch_typ)
                plotdf = plotdf.with_columns([
                    pl.col('demand_kW_mean').sum().alias('total_demand_kW_mean'), 
                    pl.col('demand_kW_median').sum().alias('total_demand_kW_median')
                ])
                plotdf_monthly = demandtypes_mean_monthly.filter(pl.col('demand_arch_typ') == arch_typ)

                plotdf = plotdf.to_pandas()
                plotdf_monthly = plotdf_monthly.to_pandas()

                fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['total_demand_kW_mean'], name= f'{arch_typ} total annual demand GAREA_mean',
                                         mode='lines', yaxis = 'y2'))
                fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['total_demand_kW_median'], name= f'{arch_typ} total annual demand by GAREA_median',
                                         mode='lines', yaxis = 'y2'))
                # fig.add_trace(go.Scatter(x=plotdf_monthly['t_int'], y=plotdf_monthly['demand_kW_mean'], name= f'{arch_typ} demand by GAREA_mean (monthly)',
                #                          mode='lines+markers', yaxis='y2'))
                # fig.add_trace(go.Scatter(x=plotdf_monthly['t_int'], y=plotdf_monthly['demand_kW_median'], name= f'{arch_typ} demand by GAREA_median (monthly)',
                #                          mode='lines+markers', yaxis='y2'))
                fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['demand_kW_mean'], name= f'{arch_typ} demand by GAREA_mean', 
                                         mode='lines', line=dict(dash='dot'), ))
                fig.add_trace(go.Scatter(x=plotdf['t_int'], y=plotdf['demand_kW_median'], name= f'{arch_typ} demand by GAREA_median',
                                         mode='lines', line=dict(dash='dot'), ))
                fig.add_trace(go.Scatter(x=[None,], y=[None,], name= '', mode='lines', opacity = 0))

            fig.update_layout(
                title=f'Demand Profiles by Demand Arch Type (scen: {scen})',
                xaxis_title='Time',
                yaxis_title='Demand [kW]',
                yaxis2=dict(
                    title='Total Demand [kWh]',
                    overlaying='y',
                    side='right',
                    showgrid=False,
                ),
                legend_title = 'Demand Arch Type',
            )
            fig = self.add_scen_name_to_plot(fig, scen, self.visual_sett.pvalloc_scen_list[i_scen])
            fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)


            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_line_demand_by_arch_typ___{scen}.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_line_demand_by_arch_typ___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_bar_totaldemand_by_type.html (for: {scen})', self.visual_sett.log_name)
