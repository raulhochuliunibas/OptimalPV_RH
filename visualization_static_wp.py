import sys
import os as os
import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns


# GENERAL SETTINGS
class static_plotter_class:
    def __init__(self):
        self.dir_path = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'visualization_static_wpaper')
        self.scen_default_color_map = {
            'pvalloc_29nbfs_30y5_max': (200, 50, 50),
            'scenario2': (50, 200, 50),
            'scenario3': (50, 50, 200),
            'scenario4': (200, 200, 50),
        }
        self.plot_width = 8
        self.plot_height = 4

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
                    label=scen
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
                    estimator=None
                    )
        plt.xlabel('t (hours)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title('Aggregated Feed-in Loss (hourly) by Iteration')
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
                )

        plt.xlabel('Model Iterations ')
        plt.ylabel(f'Aggregated {y_label} (kWh)')
        plt.title(f'Aggregated {y_label} Model Iterations')
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
                for col_idx, x_col in enumerate(x_col_incl_list):
                    sns.histplot(
                        data=df_plot,
                        x=x_col,
                        hue='iter_round',
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
        plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=300)
        

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
                plt.title(f'{col_name} - {scen}')
                plt.legend(title='Category')
                
                # Set x-axis to show only integers
                ax = plt.gca()
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                plt.tight_layout()
                plt.show()
                # Save with unique name for each scenario-column combination
                export_file = f'{export_name}_{scen}_{col_name}.png'
                plt.savefig(os.path.join(self.dir_path, export_file), dpi=300)
                plt.close()

            
    
if __name__ == "__main__":
    plotter = static_plotter_class()


    plotter.plot_productionHOY_per_node(
        csv_file='plot_agg_line_productionHOY_per_node___export_plot_data___1scen.csv',
        scen_incl_list=['pvalloc_29nbfs_30y5_max',],
        hours_incl_list=list(range(4920, 4920 + 7*24)),
        export_name='line_PVHOY_bu_loss'
    )

    plotter.plot_productionHOY_per_node_byiter(
        csv_file='plot_agg_line_productionHOY_per_node_byiter___export_plot_data___1scen.csv',
        scen_incl_list=['pvalloc_29nbfs_30y5_max',],
        hours_incl_list=list(range(4920, 4920 + 7*24)),
        iter_incl_list=['1', '2', '3', 'end_iter'],
        export_name='line_PVHOY_bu_loss_byiter'
    )

    plotter.plot_PVproduction_line(
        csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
        scen_incl_list=['pvalloc_29nbfs_30y5_max',],
        n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
        export_name='line_PVproduction_bu_loss',
        y_col='feedin_atnode_loss_kW',
        y_label='Feed-in Loss'
    )

    plotter.plot_ind_hist_contcharact_newinst(
        csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___1scen.csv',
        scen_incl_list=['pvalloc_29nbfs_30y5_max',],
        iter_incl_list=[1,3,],
        x_col_incl_list=['FLAECHE', 'GAREA'],
        export_name='hist_contcharact_newinst_bu',
        plot_height_func = 4, 
        plot_width_func =  9,
    )

    plotter.plot_ind_line_catgcharact_newinst(
        csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___1scen.csv',
        scen_incl_list=['pvalloc_29nbfs_30y5_max',],
        iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
        x_col_incl_dict={
            'GKLAS': {
                'single-family':['1110',], 
                'multi-family':['1121', '1122', ]
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
    )
        





print('end')



# # -----------------------------------------------------
# # productionHOY_per_node_byiter
# # -----------------------------------------------------
# # file_path = os.path.join(dir_path, 'plot_agg_line_productionHOY_per_node_byiter___export_plot_data___1scen.csv')
# export_name = 'line_PVHOY_bu_loss_byiter'
# scen_incl_list = ['pvalloc_29nbfs_30y5_max', ]
# hours_incl_list = list(range(4920, 4920 + 7*24))
# iter_incl_list = [
#     '1', '2', '3', 'end_iter'
#     # '1', '2', '3', '4', '5', '6', '7',
#     # '4', '6',
#     # 'end_iter' 
#     ]

# df_HOYnode_byiter = pd.read_csv(file_path)
# df_plot = df_HOYnode_byiter.loc[
#     (df_HOYnode_byiter['scen'] == 'pvalloc_29nbfs_30y5_max') &
#     (df_HOYnode_byiter['t_int'].isin(hours_incl_list)) &
#     (df_HOYnode_byiter['iter'].isin(iter_incl_list))
# ].copy()

# plt.figure(figsize=(8, 4))

# sns.lineplot(
#     data=df_plot,
#     x='t_int',
#     y='feedin_atnode_loss_kW',
#     hue='iter',
#     linewidth=1.5,
#     estimator=None
# )

# plt.xlabel('t_int (HOY)')
# plt.ylabel('Feed-in loss at node [kW]')
# plt.title('Feed-in loss over time by iteration')
# plt.legend(title='Iteration')
# plt.tight_layout()
# plt.show()

# df_HOYnode_byiter['iter'].unique()

# # -----------------------------------------------------
# # PVproduction line plot
# # -----------------------------------------------------
# file_path = os.path.join(dir_path, 'plot_agg_line_PVproduction___export_plot_data___1scen.csv')
# export_name = 'line_PVproduction_bu_loss'
# scen_incl_list = ['pvalloc_29nbfs_30y5_max', ]
# n_iter_incl_list = [4, 5, 6, 7, 8, 9, 10, ]

# df_PVpod = pd.read_csv(file_path)
# df_plot = df_PVpod.loc[
#     (df_PVpod['scen'] == 'pvalloc_29nbfs_30y5_max') & 
#     (df_PVpod['n_iter'].isin( n_iter_incl_list ))   
#     ,:].copy()
# plt.figure(figsize=(8, 4))
# sns.lineplot(
#     data=df_plot,
#     x='n_iter',
#     y='feedin_atnode_loss_kW',
#     color=rgb_color_norm
# )
# plt.xlabel('Iteration')
# plt.ylabel('Agg. Feed-in loss (kWh)')
# plt.title('Aggregated Feed-in Loss Over Model Iterations')

# plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(dir_path, f'{export_name}.png'), dpi=300)


# -----------------------------------------------------
# plot_ind_hist_contcharact_newinst
# -----------------------------------------------------
# file_path = os.path.join(dir_path, 'plot_agg_hist_contcharact_newinst___export_plot_data___1scen.csv')
# export_name = 'hist_contcharact_newinst_bu'
# iter_incl_list = [1,3,]

# df_hist_contcharact = pd.read_csv(file_path)
# df_hist_contcharact['iter_round'].unique()
# df_plot = df_hist_contcharact.loc[
#     (df_hist_contcharact['scen'] == 'pvalloc_29nbfs_30y5_max') & 
#     (df_hist_contcharact['iter_round'].isin( iter_incl_list ))
#     ,:
# ].copy()









