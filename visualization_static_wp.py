import sys
import os as os
import numpy as np
import pandas as pd
import polars as pl
import glob

import json
import matplotlib.pyplot as plt
import seaborn as sns


# GENERAL SETTINGS
class static_plotter_class:
    def __init__(self):
        self.data_path = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data')
        self.dir_path = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'visualization_static_wpaper')
        self.scen_default_color_map = {
            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'scenario2': (50, 200, 50),
            'scenario3': (50, 50, 200),
            'scenario4': (200, 200, 50),
        }
        self.line_opacity = 0.8
        self.plot_width = 8
        self.plot_height = 4
        self.show_plt_TF = False


    def plot_productionHOY_per_node(self, 
                                    csv_file, 
                                    scen_incl_list,
                                    hours_incl_list,
                                    export_name, 
                                    
                                    plot_width_func = None,
                                    plot_height_func = None, 
                                    ):

        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        np.random.seed(42) 
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        # n_random_colors = [max(0, len(scen_not_default))]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        plt.figure(figsize=(plot_width, plot_height))

        for i, scen in enumerate(scen_incl_list):
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['t_int'].isin(hours_incl_list)),
                :
            ].copy()
            scen_color = (plot_color_map[scen][0] / 255, plot_color_map[scen][1] / 255, plot_color_map[scen][2] / 255)


            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x='t_int',
                    y='feedin_atnode_loss_kW',
                    color=scen_color,
                    label=scen,
                    alpha=self.line_opacity,
                )

        plt.xlabel('t (hours)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title('Aggregated Feed-in Loss (hourly)')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=300)


    def plot_productionHOY_per_node_byiter(self, 
                                           csv_file,
                                           scen_incl_list,
                                           hours_incl_list,
                                           iter_incl_list,
                                           export_name,
                                           plot_width_func = None,
                                             plot_height_func = None,
                                             ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        np.random.seed(42) 
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        # n_random_colors = [max(0, len(scen_not_default))]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        plt.figure(figsize=(plot_width, plot_height))

        for i, scen in enumerate(scen_incl_list):
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['t_int'].isin(hours_incl_list)) &
                (df['iter'].isin(iter_incl_list)),
                :
            ].copy()
            scen_color = (plot_color_map[scen][0] / 255, plot_color_map[scen][1] / 255, plot_color_map[scen][2] / 255)


            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x='t_int',
                    y='feedin_atnode_loss_kW',
                    hue='iter',
                    palette=sns.color_palette("viridis", n_colors=len(iter_incl_list)),
                    linewidth=1.5,
                    alpha=self.line_opacity,
                    estimator=None
                    )
        plt.xlabel('t (hours)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title('Hourly Feed-in Loss by Iteration')
        plt.legend(title='Iteration')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi= 300)    

        
    def plot_PVproduction_line(self,
                                   csv_file,
                                   scen_incl_list,
                                   n_iter_range_list,
                                   export_name,
                                   y_col, 
                                   y_label,
                                   plot_width_func = None,
                                   plot_height_func = None,
                                   ):
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        np.random.seed(42)
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        plt.figure(figsize=(plot_width, plot_height))

        for i, scen in enumerate(scen_incl_list):
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['n_iter'].isin(n_iter_range_list)),
                :
            ].copy()
            scen_color = (plot_color_map[scen][0] / 255, plot_color_map[scen][1] / 255, plot_color_map[scen][2] / 255)


            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x='n_iter',
                    y=y_col,
                    color=scen_color,
                    label=scen,
                    marker='o',
                    linewidth=1.5,
                    alpha=self.line_opacity,
                )

        plt.xlabel('Model Iterations ')
        plt.ylabel(f'Aggregated {y_label} (kWh)')
        plt.title(f'Aggregated {y_label} over Model Iterations')

        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=300)


    def plot_ind_hist_contcharact_newinst(self,
                                          csv_file,
                                          scen_incl_list,
                                          iter_incl_list,
                                          x_col_incl_list,
                                          export_name,
                                          plot_width_func = None,
                                          plot_height_func = None,
                                          ):
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)
        
        # Adjust plot width based on number of subplots
        n_cols = len(x_col_incl_list)
        plot_width = (self.plot_width * n_cols) if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        np.random.seed(42) 
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        # Create subplots side by side
        fig, axes = plt.subplots(1, n_cols, figsize=(plot_width, plot_height))
        
        # If only one column, axes is not an array, so convert it
        if n_cols == 1:
            axes = [axes]

        # Loop through scenarios (assuming one scenario for now)
        for scen in scen_incl_list:
            df_plot = df.loc[
                (df['scen'] == scen) & 
                (df['iter_round'].isin(iter_incl_list)),
                :
            ].copy()

            if not df_plot.empty:
                # Loop through each column to create side-by-side histograms
                scen_short_str = scen.split('pvalloc_')[-1]
                # replace('pvalloc_29nbfs_30y5_max', '
                for col_idx, x_col in enumerate(x_col_incl_list):
                    sns.histplot(
                        data=df_plot,
                        x=x_col,
                        hue= 'iter_round',
                        multiple='layer',
                        bins=30,
                        alpha=0.6,
                        palette=sns.color_palette("viridis", n_colors=len(iter_incl_list)),
                        ax=axes[col_idx],
                        legend=True,
                        linewidth=0.1
                    )
                    axes[col_idx].set_xlabel(f'{x_col} (m2)')
                    axes[col_idx].set_ylabel('Count')
                    # axes[col_idx].set_title(f'Histogram of {x_col} by Iteration')
                    # Update legend title
                    legend = axes[col_idx].get_legend()
                    if legend:
                        legend.set_title('Iteration')
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=300)
        plt.close()
        

    def plot_ind_line_catgcharact_newinst(self, 
                                          csv_file,
                                          scen_incl_list,
                                          iter_incl_list,
                                          x_col_incl_dict,
                                          export_name,
                                          plot_width_func = None,
                                          plot_height_func = None,
                                          ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        # Loop through each scenario
        for scen in scen_incl_list:
            # Loop through each column type (e.g., 'GKLAS', 'are_typ')
            for col_name, category_groups in x_col_incl_dict.items():
                
                # Filter data for this scenario and column
                df_scen_col = df.loc[
                    (df['scen'] == scen) & 
                    (df['col'] == col_name) &
                    (df['iter'].isin(iter_incl_list)),
                    :
                ].copy()
                
                if df_scen_col.empty:
                    continue
                
                # Create new dataframe for aggregated data
                plot_data = []
                
                # Get all categories included in the dict for this column
                all_included_categories = []
                for label, categories_list in category_groups.items():
                    all_included_categories.extend(categories_list)
                
                # For each iteration, aggregate the data
                for iter_val in df_scen_col['iter'].unique():
                    df_iter = df_scen_col[df_scen_col['iter'] == iter_val]
                    
                    # Aggregate each defined group
                    for label, categories_list in category_groups.items():
                        count_sum = df_iter[df_iter['category'].isin(categories_list)]['count'].sum()
                        plot_data.append({
                            'iter': iter_val,
                            'group': label,
                            'count': count_sum
                        })
                    
                    # Calculate "rest" for categories not in the dict
                    rest_count = df_iter[~df_iter['category'].isin(all_included_categories)]['count'].sum()
                    if rest_count > 0:
                        plot_data.append({
                            'iter': iter_val,
                            'group': 'rest',
                            'count': rest_count
                        })
                
                # Convert to DataFrame
                df_plot = pd.DataFrame(plot_data)
                
                if df_plot.empty:
                    continue
                
                # Create the plot
                plt.figure(figsize=(plot_width, plot_height))
                
                # Plot each group as a line
                for group in df_plot['group'].unique():
                    df_group = df_plot[df_plot['group'] == group]
                    sns.lineplot(
                        data=df_group,
                        x='iter',
                        y='count',
                        label=group,
                        marker='o',
                        linewidth=1.5
                    )
                
                plt.xlabel('Iteration')
                plt.ylabel('Count')
                plt.title(f'{col_name}')
                plt.legend(title='Category')
                
                # Set x-axis to show only integers
                ax = plt.gca()
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                plt.tight_layout()
                # plt.show() 
                # Save with unique name for each scenario-column combination
                export_file = f'{export_name}_{scen}_{col_name}.png'
                plt.savefig(os.path.join(self.dir_path, export_file), dpi=300)
                plt.close()


    def plot_ind_hist_contcharact_allscen(self,
                                      csv_file,
                                      scen_incl_list,
                                      iter_incl_list,
                                      x_col_incl_list,
                                      export_name,
                                      plot_hist_opacity = 0.6,
                                      plot_width_func=None,
                                      plot_height_func=None):
        # Load data
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)

        # Filter for all selected scenarios and iterations
        df_plot = df[
            (df['scen'].isin(scen_incl_list)) &
            (df['iter_round'].isin(iter_incl_list))
        ].copy()

        if df_plot.empty:
            print("No data matches the selected scenarios and iterations.")
            return

        # Adjust plot width based on number of subplots
        n_cols = len(x_col_incl_list)
        plot_width = (self.plot_width * n_cols) if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        # Generate colors for scenarios
        np.random.seed(42)
        scen_not_default = [scen for scen in scen_incl_list if scen not in self.scen_default_color_map.keys()]
        random_colors_map = {scen: tuple(np.random.rand(3)) for scen in scen_not_default}
        plot_color_map = self.scen_default_color_map.copy()
        plot_color_map.update(random_colors_map)

        # Create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(plot_width, plot_height))
        if n_cols == 1:
            axes = [axes]

        # Loop over columns (variables) for histograms
        for col_idx, x_col in enumerate(x_col_incl_list):
            ax = axes[col_idx]

            # Create a new column combining scenario and iteration for hue
            df_plot['scen_str_short'] = df_plot['scen'].apply(lambda x: x.split('pvalloc_')[-1])
            df_plot['scen_iter'] = df_plot['scen_str_short'] + '_iter' + df_plot['iter_round'].astype(str)

            # Generate a color palette for each unique scenario+iteration
            unique_scen_iter = df_plot['scen_iter'].unique()
            palette = sns.color_palette("tab10", n_colors=len(unique_scen_iter))

            sns.histplot(
                data=df_plot,
                x=x_col,
                hue='scen_iter',
                multiple='layer',
                bins=30,
                alpha=plot_hist_opacity,
                palette=palette,
                ax=ax,
                linewidth=0.1,
                legend=True
            )
            ax.set_xlabel(f'{x_col}')
            ax.set_ylabel('Count')
            ax.set_title(f'Histogram of {x_col} by Scenario and Iteration')
            # Adjust legend
            legend = ax.get_legend()
            if legend:
                legend.set_title('Scenario / Iteration')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=300)
        plt.close()
    
    
    def plot_ind_line_demand(self,
                             name_dir_export ,
                             hours_incl_list,
                             export_name,
                             plot_width_func=None,
                             plot_height_func=None):
        
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func
        
        file_path = os.path.join(self.data_path, 'pvalloc', name_dir_export)

        topo    = json.load(open(os.path.join(file_path, 'topo_egid.json')))
        npv_df  = pd.read_parquet(os.path.join(file_path, 'zMC1', 'npv_df.parquet'))
        topo_subdf_paths    = glob.glob(f'{self.data_path}/pvalloc/{name_dir_export}/topo_time_subdf/topo_subdf_*.parquet')

        sfhmfh_map_list = []
        for k,v in topo.items():
            sfhmfh_map_list.append({
                'EGID': k, 
                'sfhmfh_typ': v['gwr_info']['sfhmfh_typ'], 
                'are_typ': v['gwr_info']['are_typ'],
                'gwaerzh1': v['gwr_info']['gwaerzh1'],
                'genh1': v['gwr_info']['genh1'],
            })
        sfhmfh_map_df = pd.DataFrame(sfhmfh_map_list)
        sfhmfh_map_df['heatpump_TF'] = np.where(sfhmfh_map_df['gwaerzh1'].isin(['7410', '7411']), 'heatpump', 'no_heatpump')

        npv_df_info = npv_df.merge(sfhmfh_map_df, on='EGID', how='left')

        def get_n_egids_filtered_df(df, n, sfhmfh, are, heatpump):
            df_filt = df[
                (df['sfhmfh_typ'] == sfhmfh) &
                (df['are_typ'] == are) &
                (df['heatpump_TF'] == heatpump)
            ]
            egid_list = df_filt['EGID'].unique().tolist()[:n]
            return list(df.loc[df['EGID'].isin(egid_list), 'EGID'])
        sfh_sub_hpT = get_n_egids_filtered_df(npv_df_info, 1, 'SFH', 'Suburban', 'heatpump') 
        sfh_sub_hpF = get_n_egids_filtered_df(npv_df_info, 1, 'SFH', 'Suburban', 'no_heatpump') 
        
        filter_egids_subdf = sfh_sub_hpT + sfh_sub_hpF

        topo_subdf_list = []
        for path in topo_subdf_paths:
            topo_subdf = pl.read_parquet(path)
            topo_filtr = topo_subdf.filter(pl.col('EGID').is_in(filter_egids_subdf))
            if topo_filtr.shape[0] > 0:
                topo_subdf_list.append(topo_filtr)
                topo_subdf_list.append(topo_filtr)


        topo_subdf = pl.concat(topo_subdf_list)

        # --- first df_uid per EGID ---
        topo_subdf_first = topo_subdf.group_by('EGID').agg([
            pl.first('df_uid').alias('df_uid')
        ]).to_pandas()

        # --- convert full topo_subdf to pandas for seaborn plotting ---
        topo_subdf_pd = topo_subdf.to_pandas()

        # --- prepare two plot variants: week and full year ---
        plot_variants = [
            {"hours": hours_incl_list, "suffix": "_week"},
            {"hours": None, "suffix": "_year"}  # all hours
        ]

        for variant in plot_variants:
            hours = variant["hours"]
            suffix = variant["suffix"]

            plt.figure(figsize=(plot_width, plot_height))
            np.random.seed(42)
            n_pairs = topo_subdf_first.shape[0]
            random_colors = [tuple(np.random.rand(3)) for _ in range(n_pairs)]

            single_values_list = []

            for i, row in topo_subdf_first.iterrows():
                egid = row['EGID']
                df_uid = row['df_uid']
                sfhmfh = topo[str(egid)]['gwr_info']['sfhmfh_typ']
                are_typ = topo[str(egid)]['gwr_info']['are_typ']
                heatpump_TF = 'heatpump' if topo[str(egid)]['gwr_info']['gwaerzh1'] in ['7410', '7411'] else 'no_heatpump'

                df_plot = topo_subdf_pd.loc[
                    (topo_subdf_pd['EGID'] == egid) &
                    (topo_subdf_pd['df_uid'] == df_uid)
                ].copy()

                # optional filtering for week hours
                if hours is not None and 't_int' in df_plot.columns:
                    df_plot = df_plot.loc[df_plot['t_int'].isin(hours)]

                if df_plot.shape[0] == 0:
                    continue  # skip empty

                color = random_colors[i]

                sns.lineplot(
                    data=df_plot,
                    x='t_int' if 't_int' in df_plot.columns else np.arange(len(df_plot)),
                    y='demand_kW',
                    # color=color,
                    label=f"EGID{egid} ({sfhmfh}, {are_typ}, {heatpump_TF})",
                    alpha=self.line_opacity
                )

                # collect single values
                egid = egid
                garea = topo[str(egid)]['gwr_info']['garea']
                TotalPower = npv_df.loc[npv_df['EGID'] == egid, 'TotalPower'].values[0]
                NPV = npv_df.loc[npv_df['EGID'] == egid, 'NPV_uid'].values[0]
                total_demand_kWh = df_plot['demand_kW'].sum()
                total_pvprod_kWh = df_plot['pvprod_kW'].sum() if 'pvprod_kW' in df_plot.columns else 0

                single_values_list.append({
                    'EGID': egid,
                    'GAREA': garea,
                    'TotalPower': TotalPower,
                    'NPV': NPV,
                    'Total_Demand_kWh': total_demand_kWh,
                    'Total_PVProd_kWh': total_pvprod_kWh,
                    'sfhmfh_typ': sfhmfh,
                    'are_typ': are_typ,
                    'heatpump_TF': heatpump_TF,
                })

            # plot export
            plt.xlabel('Hour (t_int)' if hours is not None else 'Index')
            plt.ylabel('Demand (kW)')
            plt.title(f'Individual Demand Profiles {suffix.strip("_")}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir_path, f"{export_name}{suffix}.png"), dpi=300)
            plt.close() 

        # export single values
        single_values_df = pd.DataFrame(single_values_list)
        single_values_df.to_csv(os.path.join(self.dir_path, f"{export_name}_single_values.csv"), index=False)




    def get_single_values(self, 
                          name_dir_export = 'pvalloc_29nbfs_30y5_max',
                          egid_str = '1366620'
                          ):
        name_dir_export_path = os.path.join(self.data_path, 'pvalloc', name_dir_export)
        topo    = json.load(open(os.path.join(name_dir_export_path, 'topo_egid.json')))
        npv_df  = pd.read_parquet(os.path.join(name_dir_export_path, 'zMC1', 'npv_df.parquet'))

        garea = topo[egid_str]['gwr_info']['GAREA']





    # NOT WORKING PROPERLY YET   
    def plot_productionHOY_iters_hue(self, 
                                    csv_file,
                                    scen_incl_list,
                                    hours_incl_list,
                                    iter_incl_list,
                                    export_name,
                                    plot_width_func=None,
                                    plot_height_func=None):
        
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)

        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        # Filter for all scenarios and hours
        df_plot = df[
            (df['scen'].isin(scen_incl_list)) &
            (df['t_int'].isin(hours_incl_list)) &
            (df['iter'].isin(iter_incl_list))
        ]

        if df_plot.empty:
            print("No data matches the selected scenarios, hours, and iterations.")
            return

        plt.figure(figsize=(plot_width, plot_height))

        # Map colors from scen_default_color_map
        color_map = {
            scen: tuple(np.array(self.scen_default_color_map[scen])/255)
            for scen in scen_incl_list
        }

        sns.lineplot(
            data=df_plot,
            x='t_int',
            y='feedin_atnode_loss_kW',
            hue='scen',   # color = scenario
            palette=color_map,
            linewidth=1.5,
            alpha=self.line_opacity,
            estimator=None
        )

        plt.xlabel('t (hours)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title(f'Hourly Feed-in Loss - Iteration {iter_incl_list[0]}')
        plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=300)
        plt.close()

    
if __name__ == "__main__":

    png_files = glob.glob(os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'visualization_static_wpaper', '*.png'))
    for png_file in png_files:
        os.remove(png_file)

    # demand and single values
    if True:
        plotter = static_plotter_class()
        plotter.plot_ind_line_demand(
            name_dir_export='pvalloc_29nbfs_30y5_max',
            hours_incl_list=list(range(4920, 4920 + 7*24)),
            export_name='example_demand_BU',
            plot_width_func=4,
            plot_height_func=4,
        )
        plotter.get_single_values()
    
    # # BU case
    if False: 
        plotter = static_plotter_class()
        # plotter.plot_productionHOY_per_node(
        #     # csv_file='plot_agg_line_productionHOY_per_node___export_plot_data___1scen.csv',
        #     csv_file='plot_agg_line_productionHOY_per_node___export_plot_data___31scen.csv',
        #     scen_incl_list=['pvalloc_29nbfs_30y5_max',],
        #     hours_incl_list=list(range(4920, 4920 + 7*24)),
        #     export_name='line_PVHOY_bu_loss'
        # )
        plotter.plot_productionHOY_per_node_byiter(
            csv_file='plot_agg_line_productionHOY_per_node_byiter___export_plot_data___1scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max',],
            hours_incl_list=list(range(4920, 4920 + 7*24)),
            iter_incl_list=['5', '6', '7', 'end_iter'],
            export_name='line_PVHOY_bu_loss_byiter',
            plot_height_func = 4, 
            plot_width_func =  4,

        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max',],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_bu_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
            plot_height_func = 4, 
            plot_width_func = 4

        )
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___1scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_bu',
            plot_height_func = 3, 
            plot_width_func =  9,
        )
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___1scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max',],
            iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
            x_col_incl_dict={
                'GKLAS': {
                    # 'single-family':['1110',], 
                    # 'multi-family':['1121', '1122', ]
                    '1 apart.':['1110',],
                    '2 apart.':['1121', ],
                    '3+ apart.':['1122', ]
                        },
                'are_typ': {
                    'rural':['Rural',],
                    'suburban':['Suburban',],
                    'urban':['Urban',]
                            },
                'heatpump_TF': {
                    'with HP':['heatpump',],
                    'without HP':['no_heatpump',]
                            }, 
                'filter_tag': {
                    'east-west': ['eastwest_80pr', 'eastwest_70pr'],
                    'south': ['south_50pr', 'south_40pr'],
                },
            },
            export_name='line_catgcharact_newinst_bu',
            plot_height_func = 4, 
            plot_width_func = 2.5,
        )

    # all casses loss appendix
    if False: 
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sAs2p0': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sAs4p0': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sAs6p0': (200, 200, 50),

            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sBs0p4': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sBs0p6': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sBs0p8': (200, 200, 50),
            
            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sCs2p4': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sCs4p6': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 200, 50),
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=4.5,


        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max',
                'pvalloc_29nbfs_30y5_max_sAs2p0',
                'pvalloc_29nbfs_30y5_max_sAs4p0',
                'pvalloc_29nbfs_30y5_max_sAs6p0',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_A_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max',
                'pvalloc_29nbfs_30y5_max_sBs0p4',
                'pvalloc_29nbfs_30y5_max_sBs0p6',
                'pvalloc_29nbfs_30y5_max_sBs0p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_B_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max',
                'pvalloc_29nbfs_30y5_max_sCs2p4',
                'pvalloc_29nbfs_30y5_max_sCs4p6',
                'pvalloc_29nbfs_30y5_max_sCs6p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_C_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )



        # plotter.plot_ind_hist_contcharact_allscen(
        #     csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___31scen.csv',
        #     scen_incl_list=[
        #         'pvalloc_29nbfs_30y5_max',
        #         # 'pvalloc_29nbfs_30y5_max_sAs2p0',
        #         # 'pvalloc_29nbfs_30y5_max_sAs4p0',
        #         'pvalloc_29nbfs_30y5_max_sAs6p0',
        #         ],
        #     iter_incl_list=[1,3,],
        #     x_col_incl_list=['FLAECHE', 'GAREA'],
        #     export_name='hist_contcharact_newinst_A',
        #     plot_hist_opacity = 0.4,
        #     plot_width_func =  9,
        #     plot_height_func = 5,
        # )

    # all casses charac comperison appendix
    if False:
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sAs2p0': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sAs4p0': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sAs6p0': (200, 200, 50),

            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sBs0p4': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sBs0p6': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sBs0p8': (200, 200, 50),
            
            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sCs2p4': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sCs4p6': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 200, 50),
        }
        cont_charc_widht = 9
        cont_charc_height = 2
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max_sAs6p0',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_A',
            plot_width_func=cont_charc_widht,
            plot_height_func=cont_charc_height
        )      
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max_sBs0p8',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_B',
            plot_width_func=cont_charc_widht,
            plot_height_func=cont_charc_height
        )
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max_sCs6p8',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_C',
            plot_width_func=cont_charc_widht,
            plot_height_func=cont_charc_height
        )

        catg_charc_widht = 2.5
        catg_charc_height = 2.75
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max_sAs6p0',],
            iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
            x_col_incl_dict={
                'GKLAS': {
                    # 'single-family':['1110',], 
                    # 'multi-family':['1121', '1122', ]
                    '1 apart.':['1110',],
                    '2 apart.':['1121', ],
                    '3+ apart.':['1122', ]
                        },
                'are_typ': {
                    'rural':['Rural',],
                    'suburban':['Suburban',],
                    'urban':['Urban',]
                            },
                'heatpump_TF': {
                    'with HP':['heatpump',],
                    'without HP':['no_heatpump',]
                            }, 
                'filter_tag': {
                    'east-west': ['eastwest_80pr', 'eastwest_70pr'],
                    'south': ['south_50pr', 'south_40pr'],
                },
            },
            export_name='line_catgcharact_newinst_A',
            plot_width_func=catg_charc_widht,
            plot_height_func=catg_charc_height
        )
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max_sBs0p8',],
            iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
            x_col_incl_dict={
                'GKLAS': {
                    # 'single-family':['1110',], 
                    # 'multi-family':['1121', '1122', ]
                    '1 apart.':['1110',],
                    '2 apart.':['1121', ],
                    '3+ apart.':['1122', ]
                        },
                'are_typ': {
                    'rural':['Rural',],
                    'suburban':['Suburban',],
                    'urban':['Urban',]
                            },
                'heatpump_TF': {
                    'with HP':['heatpump',],
                    'without HP':['no_heatpump',]
                            }, 
                'filter_tag': {
                    'east-west': ['eastwest_80pr', 'eastwest_70pr'],
                    'south': ['south_50pr', 'south_40pr'],
                },
            },
            export_name='line_catgcharact_newinst_B',
            plot_width_func=catg_charc_widht,
            plot_height_func=catg_charc_height
        )
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___31scen.csv',
            scen_incl_list=['pvalloc_29nbfs_30y5_max_sCs6p8',],
            iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
            x_col_incl_dict={
                'GKLAS': {
                    # 'single-family':['1110',], 
                    # 'multi-family':['1121', '1122', ]
                    '1 apart.':['1110',],
                    '2 apart.':['1121', ],
                    '3+ apart.':['1122', ]
                        },
                'are_typ': {
                    'rural':['Rural',],
                    'suburban':['Suburban',],
                    'urban':['Urban',]
                            },
                'heatpump_TF': {
                    'with HP':['heatpump',],
                    'without HP':['no_heatpump',]
                            }, 
                'filter_tag': {
                    'east-west': ['eastwest_80pr', 'eastwest_70pr'],
                    'south': ['south_50pr', 'south_40pr'],
                },
            },
            export_name='line_catgcharact_newinst_C',
            plot_width_func=catg_charc_widht,
            plot_height_func=catg_charc_height
        )

    # all casses production appendix
    if False:
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            'pvalloc_29nbfs_30y5_max': (180, 60, 60),        # muted red
            # --- Scheme A (greens / yellow-green gradient) ---
            'pvalloc_29nbfs_30y5_max_sAs2p0': (60, 150, 90),   # teal-green
            'pvalloc_29nbfs_30y5_max_sAs4p0': (90, 180, 60),   # green
            'pvalloc_29nbfs_30y5_max_sAs6p0': (180, 180, 60),  # yellow-green
            # --- Scheme B (blues / cyan gradient) ---
            'pvalloc_29nbfs_30y5_max_sBs0p4': (70, 130, 180),  # steel blue
            'pvalloc_29nbfs_30y5_max_sBs0p6': (60, 160, 200),  # cyan-blue
            'pvalloc_29nbfs_30y5_max_sBs0p8': (40, 190, 190),  # turquoise
            # --- Scheme C (purple / magenta gradient) ---
            'pvalloc_29nbfs_30y5_max_sCs2p4': (140, 90, 180),  # soft purple
            'pvalloc_29nbfs_30y5_max_sCs4p6': (170, 80, 150),  # magenta-purple
            'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 70, 120),  # rose-magenta
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=4.5,


        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max',
                # 'pvalloc_29nbfs_30y5_max_sAs2p0',
                # 'pvalloc_29nbfs_30y5_max_sAs4p0',
                'pvalloc_29nbfs_30y5_max_sAs6p0',
                
                # 'pvalloc_29nbfs_30y5_max_sBs0p4',
                # 'pvalloc_29nbfs_30y5_max_sBs0p6',
                'pvalloc_29nbfs_30y5_max_sBs0p8',
                
                # 'pvalloc_29nbfs_30y5_max_sCs2p4',
                # 'pvalloc_29nbfs_30y5_max_sCs4p6',
                'pvalloc_29nbfs_30y5_max_sCs6p8',
                
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABC_feedin',
            y_col='feedin_atnode_kW',
            y_label='Production',
        )



    # comparison loss max cases
    comparison_PVproduction_height = 4
    if False: 
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            # 'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_30y5_max_sAs2p0': (50, 200, 50),
            # # 'pvalloc_29nbfs_30y5_max_sAs4p0': (50, 50, 200),
            # 'pvalloc_29nbfs_30y5_max_sAs6p0': (200, 200, 50),

            # 'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_30y5_max_sBs0p4': (50, 200, 50),
            # # 'pvalloc_29nbfs_30y5_max_sBs0p6': (50, 50, 200),
            # 'pvalloc_29nbfs_30y5_max_sBs0p8': (200, 200, 50),
            
            # 'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_30y5_max_sCs2p4': (50, 200, 50),
            # # 'pvalloc_29nbfs_30y5_max_sCs4p6': (50, 50, 200),
            # 'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 200, 50),

            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_sAs6p0': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_sBs0p8': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 200, 50),
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=comparison_PVproduction_height,

        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max',
                'pvalloc_29nbfs_30y5_max_sAs6p0',
                'pvalloc_29nbfs_30y5_max_sBs0p8',
                'pvalloc_29nbfs_30y5_max_sCs6p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABC_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )

    # comparison loss 1hll cases
    if False: 
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            # 'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_30y5_max_sAs2p0': (50, 200, 50),
            # # 'pvalloc_29nbfs_30y5_max_sAs4p0': (50, 50, 200),
            # 'pvalloc_29nbfs_30y5_max_sAs6p0': (200, 200, 50),

            # 'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_30y5_max_sBs0p4': (50, 200, 50),
            # # 'pvalloc_29nbfs_30y5_max_sBs0p6': (50, 50, 200),
            # 'pvalloc_29nbfs_30y5_max_sBs0p8': (200, 200, 50),
            
            # 'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_30y5_max_sCs2p4': (50, 200, 50),
            # # 'pvalloc_29nbfs_30y5_max_sCs4p6': (50, 50, 200),
            # 'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 200, 50),

            'pvalloc_29nbfs_30y5_max_1hll': (200, 50, 50),
            'pvalloc_29nbfs_30y5_max_1hll_sAs6p0': (50, 200, 50),
            'pvalloc_29nbfs_30y5_max_1hll_sBs0p8': (50, 50, 200),
            'pvalloc_29nbfs_30y5_max_1hll_sCs4p6': (200, 200, 50),
            'pvalloc_29nbfs_30y5_max_1hll_sCs6p8': (200, 200, 50),
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=comparison_PVproduction_height,

        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max_1hll',
                'pvalloc_29nbfs_30y5_max_1hll_sAs6p0',
                'pvalloc_29nbfs_30y5_max_1hll_sBs0p8',
                'pvalloc_29nbfs_30y5_max_1hll_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABC_loss_1hll',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )


    # compmarison production 1hll cases 
    if False:
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            
            'pvalloc_29nbfs_30y5_max': (180, 60, 60),        # muted red
            'pvalloc_29nbfs_30y5_max_1hll': (220, 100, 100),    # light red

            # --- Scheme A (greens / yellow-green gradient) ---
            # 'pvalloc_29nbfs_30y5_max_sAs2p0_1hll': (60, 150, 90),   # teal-green
            'pvalloc_29nbfs_30y5_max_sAs6p0_': (90, 180, 60),   # green
            'pvalloc_29nbfs_30y5_max_1hll_sAs6p0': (180, 180, 60),  # yellow-green
            # --- Scheme B (blues / cyan gradient) ---
            # 'pvalloc_29nbfs_30y5_max_sBs0p4_1hll': (70, 130, 180),  # steel blue
            'pvalloc_29nbfs_30y5_max_sBs0p8': (60, 160, 200),  # cyan-blue
            'pvalloc_29nbfs_30y5_max_1hll_sBs0p8': (40, 190, 190),  # turquoise
            # --- Scheme C (purple / magenta gradient) ---
            # 'pvalloc_29nbfs_30y5_max_sCs2p4_1hll': (200, 70, 120),  # rose-magenta
            'pvalloc_29nbfs_30y5_max_sCs4p6': (170, 80, 150),  # magenta-purple
            'pvalloc_29nbfs_30y5_max_1hll_sCs4p6': (140, 90, 180),  # soft purple
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=comparison_PVproduction_height,


        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_30y5_max',
                'pvalloc_29nbfs_30y5_max_1hll',
                # 'pvalloc_29nbfs_30y5_max_sAs2p0',
                # 'pvalloc_29nbfs_30y5_max_sAs4p0',
                'pvalloc_29nbfs_30y5_max_sAs6p0',
                'pvalloc_29nbfs_30y5_max_1hll_sAs6p0',
                
                # 'pvalloc_29nbfs_30y5_max_sBs0p4',
                # 'pvalloc_29nbfs_30y5_max_sBs0p6',
                'pvalloc_29nbfs_30y5_max_sBs0p8',
                'pvalloc_29nbfs_30y5_max_1hll_sBs0p8',
                
                # 'pvalloc_29nbfs_30y5_max_sCs2p4',
                'pvalloc_29nbfs_30y5_max_sCs4p6',
                'pvalloc_29nbfs_30y5_max_1hll_sCs4p6',
                # 'pvalloc_29nbfs_30y5_max_sCs6p8',
                
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='buAC_feedin_w+wo1hll_line',
            y_col='feedin_atnode_kW',
            y_label='Production',
        )


#     plotter = static_plotter_class()
#     plotter.scen_default_color_map = {
#         'pvalloc_29nbfs_30y5_max': (200, 50, 50),
#         'pvalloc_29nbfs_30y5_max_sCs2p4': (50, 200, 50),
#         'pvalloc_29nbfs_30y5_max_sCs4p6': (50, 50, 200),
#         'pvalloc_29nbfs_30y5_max_sCs6p8': (200, 200, 50),
#     }
#     plotter.line_opacity = 0.6
#     # plotter.plot_productionHOY_per_node_byiter(
#     #     csv_file='plot_agg_line_productionHOY_per_node_byiter___export_plot_data___31scen.csv',
#     #     scen_incl_list=[
#     #         'pvalloc_29nbfs_30y5_max',
#     #         'pvalloc_29nbfs_30y5_max_sCs2p4',
#     #         'pvalloc_29nbfs_30y5_max_sCs4p6',
#     #         'pvalloc_29nbfs_30y5_max_sCs6p8',
#     #                     ],
#     #     hours_incl_list=list(range(4920, 4920 + 7*24)),
#     #     iter_incl_list=['5', 'end_iter'],
#     #     export_name='line_PVHOY_C_loss_byiter'
#     # )
#     # plotter.plot_productionHOY_per_node(
#     #     # csv_file='plot_agg_line_productionHOY_per_node___export_plot_data
#     #     csv_file='plot_agg_line_productionHOY_per_node___export_plot_data___31scen.csv',
#     #     scen_incl_list=[
#     #         'pvalloc_29nbfs_30y5_max',
#     #         'pvalloc_29nbfs_30y5_max_sCs2p4',
#     #         'pvalloc_29nbfs_30y5_max_sCs4p6',
#     #         'pvalloc_29nbfs_30y5_max_sCs6p8',
#     #                     ],
#     #     hours_incl_list=list(range(4920, 4920 + 7*24)),
#     #     export_name='line_PVHOY_C_loss'
#     # )
#     plotter.plot_PVproduction_line(
#         # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
#         csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
#         scen_incl_list=[
#             'pvalloc_29nbfs_30y5_max',
#             'pvalloc_29nbfs_30y5_max_sCs2p4',
#             'pvalloc_29nbfs_30y5_max_sCs4p6',
#             'pvalloc_29nbfs_30y5_max_sCs6p8',
#             ],
#         n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
#         export_name='line_PVproduction_C_loss',
#         y_col='feedin_atnode_loss_kW',
#         y_label='Feed-in Loss'
#     )

#     plotter = static_plotter_class()
#     plotter.plot_height = 6 
#     plotter.plot_width = 4

#     plotter.scen_default_color_map = {
#         'pvalloc_29nbfs_30y5_max': (200, 50, 50),
#         'pvalloc_29nbfs_30y5_max_sAs6p0': (50, 200, 50),
#         'pvalloc_29nbfs_30y5_max_sBs0p8': (50, 50, 200),
#         'pvalloc_29nbfs_30y5_max_sCs4p6': (200, 200, 50),
#         'pvalloc_29nbfs_30y5_max_sCs6p8': (150, 150, 50),

#         'pvalloc_29nbfs_30y5_max_1hll': (150, 50, 200),
#         'pvalloc_29nbfs_30y5_max_1hll_sAs6p0': (50, 200, 150),
#         'pvalloc_29nbfs_30y5_max_1hll_sBs0p8': (50, 150, 200),
#         'pvalloc_29nbfs_30y5_max_1hll_sCs4p6': (200, 150, 50),
#         'pvalloc_29nbfs_30y5_max_1hll_sCs6p8': (150, 200, 50),
    
#     }



#     plotter.plot_PVproduction_line(
#         # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
#         csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
#         scen_incl_list=[
#             'pvalloc_29nbfs_30y5_max',
#             'pvalloc_29nbfs_30y5_max_sAs6p0',
#             'pvalloc_29nbfs_30y5_max_sBs0p8',
#             'pvalloc_29nbfs_30y5_max_sCs4p6',
#             'pvalloc_29nbfs_30y5_max_sCs6p8',
#             ],
#         n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
#         export_name='line_PVproduction_bu_ABC_loss',
#         y_col='feedin_atnode_loss_kW',
#         y_label='Feed-in Loss'
#     )
#     plotter.plot_PVproduction_line(
#         # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
#         csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
#         scen_incl_list=[
#             'pvalloc_29nbfs_30y5_max_1hll',
#             'pvalloc_29nbfs_30y5_max_1hll_sAs6p0',
#             'pvalloc_29nbfs_30y5_max_1hll_sBs0p8',
#             'pvalloc_29nbfs_30y5_max_1hll_sCs4p6',
#             'pvalloc_29nbfs_30y5_max_1hll_sCs6p8',
#             ],
#         n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
#         export_name='line_PVproduction_bu_ABC_loss_1hll',
#         y_col='feedin_atnode_loss_kW',
#         y_label='Feed-in Loss'
#     )


#     plotter.plot_PVproduction_line(
#         # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
#         csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
#         scen_incl_list=[
#             'pvalloc_29nbfs_30y5_max',
#             'pvalloc_29nbfs_30y5_max_sAs6p0',
#             'pvalloc_29nbfs_30y5_max_sBs0p8',
#             'pvalloc_29nbfs_30y5_max_sCs4p6',
#             'pvalloc_29nbfs_30y5_max_sCs6p8',
#             ],
#         n_iter_range_list=[ 4, 5, ],
#         export_name='line_PVproduction_bu_ABC_loss_start',
#         y_col='feedin_atnode_loss_kW',
#         y_label='Feed-in Loss'
#     )
#     plotter.plot_PVproduction_line(
#         # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
#         csv_file='plot_agg_line_PVproduction___export_plot_data___31scen.csv',
#         scen_incl_list=[
#             'pvalloc_29nbfs_30y5_max_1hll',
#             'pvalloc_29nbfs_30y5_max_1hll_sAs6p0',
#             'pvalloc_29nbfs_30y5_max_1hll_sBs0p8',
#             'pvalloc_29nbfs_30y5_max_1hll_sCs4p6',
#             'pvalloc_29nbfs_30y5_max_1hll_sCs6p8',
#             ],
#         n_iter_range_list=[4, 5, ],
#         export_name='line_PVproduction_bu_ABC_loss_1hll_start',
#         y_col='feedin_atnode_loss_kW',
#         y_label='Feed-in Loss'
#     )




        







print('end')



#     # LRG_final_scen_list = [
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max', 
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #     ),  


#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sAs2p0',
#     #             GRIDspec_subsidy_name             = 'As2p0',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs2p0',
#     #             GRIDspec_node_1hll_closed_TF      = True,
#     #             GRIDspec_subsidy_name             = 'As2p0',
#     #     ),

#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sAs4p0',
#     #             GRIDspec_subsidy_name             = 'As4p0',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs4p0',
#     #             GRIDspec_node_1hll_closed_TF      = True,
#     #             GRIDspec_subsidy_name             = 'As4p0',
#     #     ),

#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sAs6p0',
#     #             GRIDspec_subsidy_name             = 'As6p0',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs6p0',
#     #             GRIDspec_node_1hll_closed_TF      = True,
#     #             GRIDspec_subsidy_name             = 'As6p0',
#     #     ),


#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p4',
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Bs0p4',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p4',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Bs0p4',
#     #     ),

#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p6',
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Bs0p6',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p6',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Bs0p6',
#     #     ),

#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p8',
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Bs0p8',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p8',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Bs0p8',
#     #     ),


#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p4',
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Cs2p4',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p4',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Cs2p4',
#     #     ),

#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p6',
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Cs4p6',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p6',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Cs4p6',
#     #     ),

#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sCs6p8',
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Cs6p8',
#     #     ),
#     #     make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p8',
#     #                     GRIDspec_node_1hll_closed_TF      = True,
#     #                     GRIDspec_apply_prem_tiers_TF      = True,
#     #                     GRIDspec_subsidy_name             = 'Cs6p8',
#     #     ),



# # # -----------------------------------------------------
# # # productionHOY_per_node_byiter
# # # -----------------------------------------------------
# # # file_path = os.path.join(dir_path, 'plot_agg_line_productionHOY_per_node_byiter___export_plot_data___1scen.csv')
# # export_name = 'line_PVHOY_bu_loss_byiter'
# # scen_incl_list = ['pvalloc_29nbfs_30y5_max', ]
# # hours_incl_list = list(range(4920, 4920 + 7*24))
# # iter_incl_list = [
# #     '1', '2', '3', 'end_iter'
# #     # '1', '2', '3', '4', '5', '6', '7',
# #     # '4', '6',
# #     # 'end_iter' 
# #     ]

# # df_HOYnode_byiter = pd.read_csv(file_path)
# # df_plot = df_HOYnode_byiter.loc[
# #     (df_HOYnode_byiter['scen'] == 'pvalloc_29nbfs_30y5_max') &
# #     (df_HOYnode_byiter['t_int'].isin(hours_incl_list)) &
# #     (df_HOYnode_byiter['iter'].isin(iter_incl_list))
# # ].copy()

# # plt.figure(figsize=(8, 4))

# # sns.lineplot(
# #     data=df_plot,
# #     x='t_int',
# #     y='feedin_atnode_loss_kW',
# #     hue='iter',
# #     linewidth=1.5,
# #     estimator=None
# # )

# # plt.xlabel('t_int (HOY)')
# # plt.ylabel('Feed-in loss at node [kW]')
# # plt.title('Feed-in loss over time by iteration')
# # plt.legend(title='Iteration')
# # plt.tight_layout()
# # plt.show()

# # df_HOYnode_byiter['iter'].unique()

# # # -----------------------------------------------------
# # # PVproduction line plot
# # # -----------------------------------------------------
# # file_path = os.path.join(dir_path, 'plot_agg_line_PVproduction___export_plot_data___1scen.csv')
# # export_name = 'line_PVproduction_bu_loss'
# # scen_incl_list = ['pvalloc_29nbfs_30y5_max', ]
# # n_iter_incl_list = [4, 5, 6, 7, 8, 9, 10, ]

# # df_PVpod = pd.read_csv(file_path)
# # df_plot = df_PVpod.loc[
# #     (df_PVpod['scen'] == 'pvalloc_29nbfs_30y5_max') & 
# #     (df_PVpod['n_iter'].isin( n_iter_incl_list ))   
# #     ,:].copy()
# # plt.figure(figsize=(8, 4))
# # sns.lineplot(
# #     data=df_plot,
# #     x='n_iter',
# #     y='feedin_atnode_loss_kW',
# #     color=rgb_color_norm
# # )
# # plt.xlabel('Iteration')
# # plt.ylabel('Agg. Feed-in loss (kWh)')
# # plt.title('Aggregated Feed-in Loss Over Model Iterations')

# # plt.tight_layout()
# # plt.show()
# # plt.savefig(os.path.join(dir_path, f'{export_name}.png'), dpi=300)


# # -----------------------------------------------------
# # plot_ind_hist_contcharact_newinst
# # -----------------------------------------------------
# # file_path = os.path.join(dir_path, 'plot_agg_hist_contcharact_newinst___export_plot_data___1scen.csv')
# # export_name = 'hist_contcharact_newinst_bu'
# # iter_incl_list = [1,3,]

# # df_hist_contcharact = pd.read_csv(file_path)
# # df_hist_contcharact['iter_round'].unique()
# # df_plot = df_hist_contcharact.loc[
# #     (df_hist_contcharact['scen'] == 'pvalloc_29nbfs_30y5_max') & 
# #     (df_hist_contcharact['iter_round'].isin( iter_incl_list ))
# #     ,:
# # ].copy()









