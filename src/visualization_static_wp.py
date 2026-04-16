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
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'scenario2': (50, 200, 50),
            'scenario3': (50, 50, 200),
            'scenario4': (200, 200, 50),
        }
        self.line_opacity = 0.8
        self.plot_width = 8
        self.plot_height = 4
        self.show_plt_TF = False


    def write_latex_from_template(self,
                                template_file,
                                export_file,
                                replacements):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = template_file if os.path.isabs(template_file) else os.path.join(script_dir, template_file)
        export_path = export_file if os.path.isabs(export_file) else os.path.join(self.dir_path, export_file)

        with open(template_path, 'r', encoding='utf-8') as f:
            template_text = f.read()

        filled_text = template_text.format(**replacements)

        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(filled_text)

        print(f'LaTeX file written: {export_path}')


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

        plt.xlabel('t (hours of year)')
        plt.ylabel('Feed-in loss at node (kW)')
        plt.title('Aggregated Feed-in Loss (hourly)')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=500)


    def plot_productionHOY_per_node_byiter(self, 
                                           csv_file,
                                           scen_incl_list,
                                           hours_incl_list,
                                           iter_incl_list,
                                           export_name,
                                           daynightbands = None,
                                           plot_width_func = None,
                                             plot_height_func = None,
                                             ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)
        plot_width = self.plot_width if plot_width_func is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        band_settings = None
        if daynightbands is True:
            band_settings = {}
        elif isinstance(daynightbands, dict):
            band_settings = daynightbands.copy()

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
                    linewidth=0.9,
                    alpha=self.line_opacity,
                    estimator=None
                    )

        if band_settings is not None:
            day_start_hour = int(band_settings.get('day_start_hour', 7))
            day_end_hour = int(band_settings.get('day_end_hour', 19))
            day_color = band_settings.get('day_color', '#fff7cc')
            night_color = band_settings.get('night_color', '#e6f0ff')
            band_alpha = float(band_settings.get('alpha', 0.25))

            if day_start_hour < day_end_hour:
                df_bands = df.loc[
                    (df['scen'].isin(scen_incl_list)) &
                    (df['t_int'].isin(hours_incl_list)) &
                    (df['iter'].isin(iter_incl_list)),
                    :
                ]

                if not df_bands.empty:
                    t_min = int(df_bands['t_int'].min())
                    t_max = int(df_bands['t_int'].max())

                    def is_day_hour(t_val):
                        hour_of_day = ((int(t_val) - 1) % 24) + 1
                        return day_start_hour <= hour_of_day < day_end_hour

                    ax = plt.gca()
                    segment_start = t_min
                    prev_is_day = is_day_hour(t_min)

                    for t_val in range(t_min + 1, t_max + 1):
                        curr_is_day = is_day_hour(t_val)
                        if curr_is_day != prev_is_day:
                            segment_color = day_color if prev_is_day else night_color
                            ax.axvspan(segment_start - 0.5, t_val - 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)
                            segment_start = t_val
                            prev_is_day = curr_is_day

                    segment_color = day_color if prev_is_day else night_color
                    ax.axvspan(segment_start - 0.5, t_max + 0.5, color=segment_color, alpha=band_alpha, zorder=0, linewidth=0)

                    for line in ax.lines:
                        line.set_zorder(3)

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

            # unit conversion (kWh to MWh)
            df_plot[y_col] = df_plot[y_col] / 1000
            scen_label = scen.split('pvalloc_29nbfs_')[-1]
            if y_col == 'feedin_atnode_taken_kW':
                df_plot[y_col] = df_plot[y_col] / 1000




            if not df_plot.empty:
                sns.lineplot(
                    data=df_plot,
                    x='n_iter',
                    y=y_col,
                    color=scen_color,
                    # label=scen,
                    label=scen_label,
                    marker='o',
                    linewidth=1.5,
                    alpha=self.line_opacity,
                )

        plt.xlabel('Model Iterations (Future Years)')
        if y_col == 'feedin_atnode_taken_kW':
            plt.ylabel(f'Aggregated {y_label} (GWh)')
        else:
            plt.ylabel(f'Aggregated {y_label} (MWh)')
        plt.title(f'Agg. {y_label}')

        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=500)


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
                # replace('pvalloc_29nbfs_LRG2_max', '
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
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=500)
        plt.close()
        

    def plot_ind_line_catgcharact_newinst(self, 
                                          csv_file,
                                          scen_incl_list,
                                        #   iter_incl_list,
                                          x_col_incl_dict,
                                          export_name,
                                          iter_incl_list=[1, 2, 3, 4, 5, 6, 7,  ],
                                          crop_right_x = None,
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
                plt.legend(title = None)
                
                # Set x-axis to show only integers
                ax = plt.gca()
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                if crop_right_x is not None:
                    ax.set_xlim(right=crop_right_x)
                
                plt.tight_layout()
                # plt.show() 
                # Save with unique name for each scenario-column combination
                export_file = f'{export_name}_{scen}_{col_name}.png'
                plt.savefig(os.path.join(self.dir_path, export_file), dpi=500)
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
        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=500)
        plt.close()
    
    
    def plot_ind_line_demand(self,
                             name_dir_export ,
                             hours_incl_list,
                             export_name,
                             select_egids = None,
                             n_egids_by_group = {
                                 'sfh_rur_hpF': (0, 'SFH', 'Rural',     'no_heatpump'),
                                 'sfh_rur_hpT': (0, 'SFH', 'Rural',     'heatpump'),
                                 'sfh_sub_hpF': (0, 'SFH', 'Suburban',  'no_heatpump'),
                                 'sfh_sub_hpT': (0, 'SFH', 'Suburban',  'heatpump'),
                                 'sfh_urb_hpF': (0, 'SFH', 'Urban',     'no_heatpump'),
                                 'sfh_urb_hpT': (0, 'SFH', 'Urban',     'heatpump'),
                                 'mfh_rur_hpF': (0, 'MFH', 'Rural',     'no_heatpump'),
                                 'mfh_rur_hpT': (0, 'MFH', 'Rural',     'heatpump'),
                                 'mfh_sub_hpF': (0, 'MFH', 'Suburban',  'no_heatpump'),
                                 'mfh_sub_hpT': (0, 'MFH', 'Suburban',  'heatpump'),
                                 'mfh_urb_hpF': (0, 'MFH', 'Urban',     'no_heatpump'),
                                 'mfh_urb_hpT': (0, 'MFH', 'Urban',     'heatpump'),
                             },
                             export_plots = True, 
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
        n_egids_list = []

        for k,v in n_egids_by_group.items():
            n_egids, sfhmfh, are, heatpump = v
            if n_egids > 0:
                get_egids = get_n_egids_filtered_df(npv_df_info, n_egids, sfhmfh, are, heatpump)
                n_egids_list.extend(get_egids)

        if select_egids is not None and len(select_egids) > 0:
            filter_egids_subdf = select_egids
        else:
            filter_egids_subdf = n_egids_list

        # sfh_sub_hpT = get_n_egids_filtered_df(npv_df_info, 10, 'SFH', 'Suburban', 'heatpump') 
        # sfh_sub_hpF = get_n_egids_filtered_df(npv_df_info, 0, 'SFH', 'Suburban', 'no_heatpump')         
        # filter_egids_subdf = sfh_sub_hpT + sfh_sub_hpF

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

                if export_plots: 
                    sns.lineplot(
                        data=df_plot,
                        x='t_int' if 't_int' in df_plot.columns else np.arange(len(df_plot)),
                        y='demand_kW',
                        # color=color,
                        # label=f"EGID {egid} ({sfhmfh}, {are_typ}, {heatpump_TF})",
                        label=f"{sfhmfh}, {are_typ}, {heatpump_TF}",
                        alpha=self.line_opacity,
                        linewidth=1.5,
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

            if export_plots:
                # plot export
                plt.xlabel('Hour (t_int)' if hours is not None else 'Index')
                plt.ylabel('Demand (kW)')
                plt.title(f'Individual Demand Profiles ({suffix.strip("_")})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.dir_path, f"{export_name}{suffix}.png"), dpi=500)
                plt.close() 

        # export single values
        single_values_df = pd.DataFrame(single_values_list)
        single_values_df.to_csv(os.path.join(self.dir_path, f"{export_name}_single_values.csv"), index=False)




    def get_single_values(self, 
                          csv_file,
                          scen_incl_list,
                          n_iter_range_list,
 ):
        
        file_path = os.path.join(self.dir_path, csv_file)
        df = pd.read_csv(file_path)



        df_comp = df.loc[
            (df['scen'].isin(scen_incl_list
                )) & 
            (df['n_iter'].isin(n_iter_range_list
                )),
            :].copy()
        
        for scen in scen_incl_list:

            # print losses
            scen_loss = df_comp.loc[df_comp['scen'] == scen, 'feedin_atnode_loss_kW'].values[0]
            print_str = f'{scen} loss: {round(scen_loss,1)} kWh/y'
            print(f'\n{print_str}\n')
            
            with open (os.path.join(self.dir_path, "loss_comparison_single_values.txt"), 'a') as f:
                f.write(f'{print_str}\n')

            # # print topo egid count
            # topo_path = os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'pvalloc', scen , 'topo_egid.json')
            # topo = json.load(open(topo_path))

            # print_str = f'{scen} topo_egid_count: {len(topo)}'
            # print(f'\n{print_str}\n')
            # with open (os.path.join(self.dir_path, "loss_comparison_single_values.txt"), 'a') as f:
            #     f.write(f'{print_str}\n')

    def SustFuturePresent_miscellaneous(self, 
                                        scen = 'pvalloc_29nbfs_LRG2_max',
                                        npv_hist_width = 3.85,
                                        npv_hist_height= 3.4,
                                        npv_hist_xrange = (-1e4, 4.75e4),
                                        ):
        # get npv_df histogramm
        if True: 
            npv_df = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'npv_df.parquet'))

            if 'NPV_uid_before_subsidy' not in npv_df.columns:
                print("Column 'NPV_uid_before_subsidy' not found in npv_df.")
                return

            df_plot = npv_df.loc[npv_df['NPV_uid_before_subsidy'].notna(), ['NPV_uid_before_subsidy']].copy()
            if df_plot.empty:
                print("No values found in 'NPV_uid_before_subsidy' for histogram.")
                return

            mean_val = df_plot['NPV_uid_before_subsidy'].mean()
            median_val = df_plot['NPV_uid_before_subsidy'].median()
            viridis_palette = sns.color_palette("viridis", n_colors=6)

            plt.figure(figsize=(npv_hist_width, npv_hist_height))
            ax = sns.histplot(
                data=df_plot,
                x='NPV_uid_before_subsidy',
                bins=40,
                kde=False,
                color=viridis_palette[3],
                alpha=0.75,
                linewidth=0.1
            )
            ax.axvline(mean_val, color=viridis_palette[2], linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:,.0f}')
            ax.axvline(median_val, color=viridis_palette[1], linestyle='-', linewidth=1.5, label=f'Median: {median_val:,.0f}')
            if len(npv_hist_xrange) == 2:
                ax.set_xlim(npv_hist_xrange[0], npv_hist_xrange[1])
            plt.xlabel('NPV before subsidy (CHF)')
            plt.ylabel('Count')
            plt.title('Distribution of NPV before Subsidy')
            plt.legend(title=None)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(self.dir_path, f'{scen}_npv_df_hist.png'), dpi=500)
            plt.close()
        

        # get data for data table

        if True:
            topo = json.load(open(os.path.join(self.data_path, 'pvalloc', scen, 'topo_egid.json')))
            
            topo_gwr_summary_list = []
            for k, v in topo.items():
                topo_gwr_summary_list.append({
                    'EGID':           k,
                    'grid_node':      v['grid_node'],
                    'bfs':            v['gwr_info']['bfs'],
                    'garea':          v['gwr_info']['garea'],
                    'heating_system': v['gwr_info']['heating_system'],
                })
            topo_gwr_summary_df = pd.DataFrame(topo_gwr_summary_list)
            
            topo_solkat_summary_list = []
            for k, v in topo.items():
                for k_solkat, v_solkat in v.get('solkat_partitions', {}).items():
                    topo_solkat_summary_list.append({
                       'EGID': k,
                       'duid': k_solkat,
                       'FLAECHE': v_solkat['FLAECHE'],
                       'STROMERTRAG': v_solkat['STROMERTRAG'],
                   }) 
            topo_solkat_summary_df = pd.DataFrame(topo_solkat_summary_list)

            constrcapa = pd.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'constrcapa.parquet'))

            n_municipalities    = topo_gwr_summary_df['bfs'].nunique()
            n_egids            = topo_gwr_summary_df.shape[0]
            n_grid_nodes        = topo_gwr_summary_df['grid_node'].nunique()
            total_roof_surface  = topo_solkat_summary_df['FLAECHE'].sum()
            capacity_mw_2050    = list(constrcapa['constr_capacity_kw'])[-1] / 1000

            # ALLDSO sample! 
            ALLDSO_bfs_list = [
                # RURAL 
                2612, 2889, 2883, 2621, 2622,
                2620, 2615, 2614, 2616, 2480,
                2617, 2611, 2788, 2619, 2783, 2477, 
                2790, 2426, 2428, 2893, 
                2885, 2852, 2491, 2502, 2499,
                2585,
                # SUBURBAN
                2613, 2782, 2618, 2786, 2785, 
                2772, 2761, 2743, 2476, 2768,
                2471, 2481, 2775, 2764, 2771, 
                2763, 2473, 2475, 2474, 2472, 
                2478, 2830, 2766, 2767, 2774, 
                2422, 2792, 2787, 2789, 2479, 
                2834, 2833, 
                2579, 2582, 2586, 2500, 2501, 
                2493, 2497, 2495, 2584, 2573, 
                2572, 2576, 2583,
                # URBAN
                2773, 2769, 2770,
                2762, 2765, 
                2829, 2831, 
                2581,
            ]

            LRG_bfs_list = [
            # RURAL 
            2612, 2889, 2883, 2621, 2622,
            2620, 2615, 2614, 2616, 2480,
            2617, 2611, 2788, 2619, 2783, 2477, 
            # SUBURBAN
            2613, 2782, 2618, 2786, 2785, 
            2772, 2761, 2743, 2476, 2768,
            # URBAN
            2773, 2769, 2770,
            ]

            preprep_gwr_all_buildilngs = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr_all_building_df.parquet")
            preprep_gwr                = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\gwr.parquet")
            dsonodes_df                = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\dsonodes_df.parquet")
            solkat                     = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\preprep\preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26\solkat.parquet")
            
            preprep_gwr['GGDENR'] = preprep_gwr['GGDENR'].astype(int)
            solkat['BFS_NUMMER'] = solkat['BFS_NUMMER'].astype(int)


            # LRG specific
            n_houses = preprep_gwr_all_buildilngs.loc[preprep_gwr_all_buildilngs['BFS_NO'].isin(LRG_bfs_list),'EGID'].nunique()
            n_egids_check = preprep_gwr.loc[preprep_gwr['GGDENR'].isin(LRG_bfs_list),'EGID'].nunique()

            # ALLDSO specific
            alldso_houses = preprep_gwr_all_buildilngs.loc[preprep_gwr_all_buildilngs['BFS_NO'].isin(ALLDSO_bfs_list),'EGID'].nunique()
            alldso_egids = preprep_gwr.loc[preprep_gwr['GGDENR'].isin(ALLDSO_bfs_list),'EGID'].nunique()
            alldso_grid_nodes = dsonodes_df['grid_node'].nunique()
            alldso_total_roof_surface = solkat.loc[solkat['BFS_NUMMER'].isin(ALLDSO_bfs_list), 'FLAECHE'].sum()



            print(f'n egids from topo: {n_egids}, from gwr: {n_egids_check}')
            def fmt(n):
                return f"{n:,}".replace(",", "'")
            replacements = {
                "n_municipalities":         fmt(n_municipalities),
                "n_houses":                 fmt(n_houses),
                "n_egids":                  fmt(n_egids),
                "n_grid_nodes":             fmt(n_grid_nodes),
                "total_roof_surface":       fmt(round(total_roof_surface, 1)),
                "capacity_mw_2050":         fmt(round(capacity_mw_2050, 1)),
                "alldso_municipalities":    fmt(len(ALLDSO_bfs_list)),
                "alldso_houses":            fmt(alldso_houses),
                "alldso_egids":             fmt(alldso_egids),
                "alldso_grid_nodes":        fmt(alldso_grid_nodes),
                "alldso_total_roof_surface": fmt(round(alldso_total_roof_surface, 1)),
                }

            plotter.write_latex_from_template(
                template_file="latex_table_template.txt",
                export_file="summary_stats.txt",
                replacements=replacements,
            )




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

        plt.savefig(os.path.join(self.dir_path, f'{export_name}.png'), dpi=500)
        plt.close()

    
if __name__ == "__main__":

    # png_files = glob.glob(os.path.join('C:',os.sep, 'Models', 'OptimalPV_RH', 'data', 'visualization_static_wpaper', '*.png'))
    # for png_file in png_files:
    #     os.remove(png_file)

    # demand and single values
    if True:
        plotter = static_plotter_class()
        # plotter.get_single_values(
        #     csv_file='plot_agg_line_PVproduction___export_plot_data___14scen.csv',
        #     scen_incl_list=[
        #     #     'pvalloc_29nbfs_LRG2_max_1hll',
        #     #     'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6',
        #            'DEV_pvalloc_29nbfs_LRG_max_OLDpreprep',
        #            'DEV_pvalloc_29nbfs_LRG_max_sCs4p6_OLDpreprep',
        #         ],
        #         n_iter_range_list=[20,],

        # )
        plotter.SustFuturePresent_miscellaneous()

        print('\n--- demand profiles ---')
        plotter.plot_ind_line_demand(
            name_dir_export='DEV_pvalloc_10nbfs_SUB_max_OLDpreprep',
            hours_incl_list=list(range(4920, 4920 + 7*24)),
            export_name='example_demand_BU',
            n_egids_by_group = {
                'sfh_rur_hpT': (1, 'SFH', 'Rural',     'heatpump'),
                'sfh_rur_hpF': (1, 'SFH', 'Rural',     'no_heatpump'),
                # 'sfh_sub_hpT': (0, 'SFH', 'Suburban',  'heatpump'),
                # 'sfh_urb_hpF': (0, 'SFH', 'Urban',     'no_heatpump'),
                # 'sfh_urb_hpT': (0, 'SFH', 'Urban',     'heatpump'),
                # 'sfh_sub_hpF': (0, 'SFH', 'Suburban',  'no_heatpump'),

                # 'mfh_rur_hpT': (1, 'MFH', 'Rural',     'heatpump'),
                # 'mfh_rur_hpF': (1, 'MFH', 'Rural',     'no_heatpump'),
                # 'mfh_sub_hpT': (0, 'MFH', 'Suburban',  'heatpump'),
                # 'mfh_sub_hpF': (0, 'MFH', 'Suburban',  'no_heatpump'),
                # 'mfh_urb_hpT': (0, 'MFH', 'Urban',     'heatpump'),
                # 'mfh_urb_hpF': (0, 'MFH', 'Urban',     'no_heatpump'),
                             },
            # select_egids = [
            #     '101221005', # MFH, Rural, heatpump
            #     '245048874', # SFH, Suburban, heatpump
            # ],
            # export_plots=False,
            plot_width_func=4,
            plot_height_func=4,
        )

    
    # BU case
    if True: 

        bu_loss_height = 5.8
        bu_loss_width = 4


        print('- plot_productionHOY_per_node')
        plotter.plot_productionHOY_per_node(
            # csv_file='plot_agg_line_productionHOY_per_node___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_productionHOY_per_node___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            hours_incl_list=list(range(4920 + 3*24, 4920 + 6*24)),
            export_name='line_PVHOY_bu_loss',
            plot_height_func = bu_loss_height,
            plot_width_func  = bu_loss_width,
        )
        plotter = static_plotter_class()
        print('\n--- BU case ---')
        print('- plot_productionHOY_per_node_byiter')
        plotter.plot_productionHOY_per_node_byiter(
            csv_file='plot_agg_line_productionHOY_per_node_byiter___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            hours_incl_list=list(range(4920 + 3*24, 4920 + 6*24)),
            iter_incl_list=['5', '6', '7', ], #'end_iter'],
            export_name='line_PVHOY_bu_loss_byiter',
            daynightbands = True,
            plot_height_func = bu_loss_height,
            plot_width_func  = bu_loss_width,
            )
        print('- plot_PVproduction_line')
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_bu_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
            plot_height_func = bu_loss_height,
            plot_width_func  = bu_loss_width,
        )
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            iter_incl_list=[1, 3, 5],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_bu',
            plot_height_func = 3, 
            plot_width_func =  8.5,
        )
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            # iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
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
                    'HP':['heatpump',],
                    'no HP':['no_heatpump',]
                            }, 
                'filter_tag': {
                    'east-west': ['eastwest_80pr', 'eastwest_70pr'],
                    'south': ['south_50pr', 'south_40pr'],
                },
            },
            export_name='line_catgcharact_newinst_bu',
            plot_height_func = 2.85, 
            # plot_width_func = 2.5,
            plot_width_func =2.4,
        )


    # ABC comparison 
    if True: 
        # comparison_PVproduction_height = 4
        # comparison_PVproduction_width = 8.5
        comparison_PVproduction_height = 5.8
        comparison_PVproduction_width = 4

        
        plotter = static_plotter_class()
        # plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # 'scenario2': (50, 200, 50), 'scenario3': (50, 50, 200), 'scenario4': (200, 200, 50),

            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # 'pvalloc_29nbfs_LRG2_max_sAs4p0': (50, 200, 50),
            # 'pvalloc_29nbfs_LRG2_max_sBs0p8': (50, 50, 200),
            # 'pvalloc_29nbfs_LRG2_max_sCs2p8': (200, 200, 50),
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),      # red (keep)
            'pvalloc_29nbfs_LRG2_max_sAs4p0': (60, 120, 200),  # strong blue
            'pvalloc_29nbfs_LRG2_max_sBs0p8': (60, 160, 90),   # green (less neon)
            'pvalloc_29nbfs_LRG2_max_sCs2p8': (200, 140, 40),  # orange (instead of yellow)
            'pvalloc_29nbfs_LRG2_max_sCs4p6': (100, 180, 180),  # turquoise (instead of yellow)
        }
        plotter.plot_width  = comparison_PVproduction_width
        plotter.plot_height = comparison_PVproduction_height

        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sAs4p0',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABC_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sAs4p0',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            # export_name='line_PVproduction_buABC_loss',
            # y_col='feedin_atnode_loss_kW',
            # y_label='Feed-in Loss',
            export_name='line_PVproduction_buABC_feedin',
            y_col='feedin_atnode_taken_kW',
            y_label='Feedin',
        )


        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sCs2p8',],
            iter_incl_list=[1, 3, 5],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_C',
            plot_height_func = 3, 
            plot_width_func =  8.5,
        )
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sCs2p8',],
            # iter_incl_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, ],
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
                    'HP':['heatpump',],
                    'no HP':['no_heatpump',]
                            }, 
                'filter_tag': {
                    'east-west': ['eastwest_80pr', 'eastwest_70pr'],
                    'south': ['south_50pr', 'south_40pr'],
                },
            },
            export_name='line_catgcharact_newinst_C',
            plot_height_func = 2.85, 
            # plot_width_func = 2.5,
            plot_width_func =2.4,
        )

    # 1HLL comparison
    if True: 
        # comparison_PVproduction_height = 4
        # comparison_PVproduction_width = 8.5
        comparison_PVproduction_height = 5.8
        comparison_PVproduction_width = 4

        
        plotter = static_plotter_class()
        # plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # 'scenario2': (50, 200, 50), 'scenario3': (50, 50, 200), 'scenario4': (200, 200, 50),

            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # 'pvalloc_29nbfs_LRG2_max_sAs4p0': (50, 200, 50),
            # 'pvalloc_29nbfs_LRG2_max_sBs0p8': (50, 50, 200),
            # 'pvalloc_29nbfs_LRG2_max_sCs2p8': (200, 200, 50),
            'pvalloc_29nbfs_LRG2_max':              (200, 50, 50),      # red (keep)
            'pvalloc_29nbfs_LRG2_max_1hll':         (230, 140, 140),  # soft pastel red
            'pvalloc_29nbfs_LRG2_max_1hll_sAs4p0':  (140, 180, 230),  # pastel blue
            'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8':  (150, 210, 170),  # pastel green
            'pvalloc_29nbfs_LRG2_max_1hll_sCs2p8':  (240, 200, 140),  # pastel orange
            'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6':  (180, 220, 220),  # pastel turquoise    
        }
        plotter.plot_width  = comparison_PVproduction_width
        plotter.plot_height = comparison_PVproduction_height

        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen_ADJ.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max', 
                'pvalloc_29nbfs_LRG2_max_1hll', 
                # 'pvalloc_29nbfs_LRG2_max_1hll_sAs4p0',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs2p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABC_1hll_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___31scen_ADJ.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max', 
                'pvalloc_29nbfs_LRG2_max_1hll', 
                # 'pvalloc_29nbfs_LRG2_max_1hll_sAs4p0',
                # 'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs2p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            # export_name='line_PVproduction_buABC_loss',
            # y_col='feedin_atnode_loss_kW',
            # y_label='Feed-in Loss',
            export_name='line_PVproduction_buAB_1hll_feedin',
            y_col='feedin_atnode_kW',
            y_label='Feedin',
        )


    # ABC + gridoptim comparison 
    if True: 
        # comparison_PVproduction_height = 4
        # comparison_PVproduction_width = 8.5
        comparison_PVproduction_height = 5.8
        comparison_PVproduction_width = 4

        
        plotter = static_plotter_class()
        # plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # 'scenario2': (50, 200, 50), 'scenario3': (50, 50, 200), 'scenario4': (200, 200, 50),

            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # 'pvalloc_29nbfs_LRG2_max_sAs4p0': (50, 200, 50),
            # 'pvalloc_29nbfs_LRG2_max_sBs0p8': (50, 50, 200),
            # 'pvalloc_29nbfs_LRG2_max_sCs2p8': (200, 200, 50),
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),      # red (keep)
            'pvalloc_29nbfs_LRG2_max_sAs4p0': (60, 120, 200),  # strong blue
            'pvalloc_29nbfs_LRG2_max_sBs0p8': (60, 160, 90),   # green (less neon)
            'pvalloc_29nbfs_LRG2_max_sCs2p8': (200, 140, 40),  # orange (instead of yellow)
            'pvalloc_29nbfs_LRG2_max_sCs4p6': (100, 180, 180),  # turquoise (instead of yellow)
            'pvalloc_29nbfs_LRG2_gridoptim_max': (120, 60, 200)      # purple

        }
        plotter.plot_width  = comparison_PVproduction_width
        plotter.plot_height = comparison_PVproduction_height

        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_gridoptim_max',
                # 'pvalloc_29nbfs_LRG2_max_sAs4p0',
                # 'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABCgridopt_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_gridoptim_max',
                # 'pvalloc_29nbfs_LRG2_max_sAs4p0',
                # 'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            # export_name='line_PVproduction_buABC_loss',
            # y_col='feedin_atnode_loss_kW',
            # y_label='Feed-in Loss',
            export_name='line_PVproduction_buABCgridopt_feedin',
            y_col='feedin_atnode_taken_kW',
            y_label='Feedin',
        )

    # comparison loss 1hll cases
    if False: 
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_LRG2_max_sAs2p0': (50, 200, 50),
            # # 'pvalloc_29nbfs_LRG2_max_sAs4p0': (50, 50, 200),
            # 'pvalloc_29nbfs_LRG2_max_sAs6p0': (200, 200, 50),

            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_LRG2_max_sBs0p4': (50, 200, 50),
            # # 'pvalloc_29nbfs_LRG2_max_sBs0p6': (50, 50, 200),
            # 'pvalloc_29nbfs_LRG2_max_sBs0p8': (200, 200, 50),
            
            # 'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            # # 'pvalloc_29nbfs_LRG2_max_sCs2p4': (50, 200, 50),
            # # 'pvalloc_29nbfs_LRG2_max_sCs4p6': (50, 50, 200),
            # 'pvalloc_29nbfs_LRG2_max_sCs6p8': (200, 200, 50),

            'pvalloc_29nbfs_LRG2_max_1hll': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_1hll_sAs6p0': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6': (200, 200, 50),
            'pvalloc_29nbfs_LRG2_max_1hll_sCs6p8': (200, 200, 50),
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=comparison_PVproduction_height,

        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max_1hll',
                'pvalloc_29nbfs_LRG2_max_1hll_sAs6p0',
                'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6',
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
            
            'pvalloc_29nbfs_LRG2_max': (180, 60, 60),        # muted red
            'pvalloc_29nbfs_LRG2_max_1hll': (220, 100, 100),    # light red

            # --- Scheme A (greens / yellow-green gradient) ---
            # 'pvalloc_29nbfs_LRG2_max_sAs2p0_1hll': (60, 150, 90),   # teal-green
            'pvalloc_29nbfs_LRG2_max_sAs6p0_': (90, 180, 60),   # green
            'pvalloc_29nbfs_LRG2_max_1hll_sAs6p0': (180, 180, 60),  # yellow-green
            # --- Scheme B (blues / cyan gradient) ---
            # 'pvalloc_29nbfs_LRG2_max_sBs0p4_1hll': (70, 130, 180),  # steel blue
            'pvalloc_29nbfs_LRG2_max_sBs0p8': (60, 160, 200),  # cyan-blue
            'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8': (40, 190, 190),  # turquoise
            # --- Scheme C (purple / magenta gradient) ---
            # 'pvalloc_29nbfs_LRG2_max_sCs2p4_1hll': (200, 70, 120),  # rose-magenta
            'pvalloc_29nbfs_LRG2_max_sCs4p6': (170, 80, 150),  # magenta-purple
            'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6': (140, 90, 180),  # soft purple
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=comparison_PVproduction_height,


        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_1hll',
                # 'pvalloc_29nbfs_LRG2_max_sAs2p0',
                # 'pvalloc_29nbfs_LRG2_max_sAs4p0',
                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                'pvalloc_29nbfs_LRG2_max_1hll_sAs6p0',
                
                # 'pvalloc_29nbfs_LRG2_max_sBs0p4',
                # 'pvalloc_29nbfs_LRG2_max_sBs0p6',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_1hll_sBs0p8',
                
                # 'pvalloc_29nbfs_LRG2_max_sCs2p4',
                'pvalloc_29nbfs_LRG2_max_sCs4p6',
                'pvalloc_29nbfs_LRG2_max_1hll_sCs4p6',
                # 'pvalloc_29nbfs_LRG2_max_sCs6p8',
                
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='buAC_feedin_w+wo1hll_line',
            y_col='feedin_atnode_taken_kW',
            y_label='Production',
        )



    # all casses loss APPENDIX
    if False: 
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_sAs2p0': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_sAs4p0': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_sAs6p0': (200, 200, 50),

            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_sBs0p4': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_sBs0p6': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_sBs0p8': (200, 200, 50),
            
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_sCs2p4': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_sCs4p6': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_sCs6p8': (200, 200, 50),
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=4.5,


        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sAs2p0',
                'pvalloc_29nbfs_LRG2_max_sAs4p0',
                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_A_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sBs0p4',
                'pvalloc_29nbfs_LRG2_max_sBs0p6',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_B_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )
        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sCs2p4',
                'pvalloc_29nbfs_LRG2_max_sCs4p6',
                'pvalloc_29nbfs_LRG2_max_sCs6p8',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_C_loss',
            y_col='feedin_atnode_loss_kW',
            y_label='Feed-in Loss',
        )


        # plotter.plot_ind_hist_contcharact_allscen(
        #     csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___17scen.csv',
        #     scen_incl_list=[
        #         'pvalloc_29nbfs_LRG2_max',
        #         # 'pvalloc_29nbfs_LRG2_max_sAs2p0',
        #         # 'pvalloc_29nbfs_LRG2_max_sAs4p0',
        #         'pvalloc_29nbfs_LRG2_max_sAs6p0',
        #         ],
        #     iter_incl_list=[1,3,],
        #     x_col_incl_list=['FLAECHE', 'GAREA'],
        #     export_name='hist_contcharact_newinst_A',
        #     plot_hist_opacity = 0.4,
        #     plot_width_func =  9,
        #     plot_height_func = 5,
        # )

    # all casses charac comperison APPENDX
    if False:
        plotter = static_plotter_class()
        plotter.line_opacity = 0.6 
        plotter.scen_default_color_map = {
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_sAs2p0': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_sAs4p0': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_sAs6p0': (200, 200, 50),

            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_sBs0p4': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_sBs0p6': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_sBs0p8': (200, 200, 50),
            
            'pvalloc_29nbfs_LRG2_max': (200, 50, 50),
            'pvalloc_29nbfs_LRG2_max_sCs2p4': (50, 200, 50),
            'pvalloc_29nbfs_LRG2_max_sCs4p6': (50, 50, 200),
            'pvalloc_29nbfs_LRG2_max_sCs6p8': (200, 200, 50),
        }
        cont_charc_widht = 9
        cont_charc_height = 2
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sAs6p0',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_A',
            plot_width_func=cont_charc_widht,
            plot_height_func=cont_charc_height
        )      
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sBs0p8',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_B',
            plot_width_func=cont_charc_widht,
            plot_height_func=cont_charc_height
        )
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file='plot_agg_hist_contcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sCs6p8',],
            iter_incl_list=[1, 4,],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_C',
            plot_width_func=cont_charc_widht,
            plot_height_func=cont_charc_height
        )

        # catg_charc_widht = 3.3
        catg_charc_widht = 4
        catg_charc_height = 2.75
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sAs6p0',],
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
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sBs0p8',],
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
            csv_file='plot_agg_bar_catgcharact_newinst___export_plot_data___17scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max_sCs6p8',],
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
            'pvalloc_29nbfs_LRG2_max': (180, 60, 60),        # muted red
            # --- Scheme A (greens / yellow-green gradient) ---
            'pvalloc_29nbfs_LRG2_max_sAs2p0': (60, 150, 90),   # teal-green
            'pvalloc_29nbfs_LRG2_max_sAs4p0': (90, 180, 60),   # green
            'pvalloc_29nbfs_LRG2_max_sAs6p0': (180, 180, 60),  # yellow-green
            # --- Scheme B (blues / cyan gradient) ---
            'pvalloc_29nbfs_LRG2_max_sBs0p4': (70, 130, 180),  # steel blue
            'pvalloc_29nbfs_LRG2_max_sBs0p6': (60, 160, 200),  # cyan-blue
            'pvalloc_29nbfs_LRG2_max_sBs0p8': (40, 190, 190),  # turquoise
            # --- Scheme C (purple / magenta gradient) ---
            'pvalloc_29nbfs_LRG2_max_sCs2p4': (140, 90, 180),  # soft purple
            'pvalloc_29nbfs_LRG2_max_sCs4p6': (170, 80, 150),  # magenta-purple
            'pvalloc_29nbfs_LRG2_max_sCs6p8': (200, 70, 120),  # rose-magenta
        }
        plotter.plot_width_func=9,
        plotter.plot_height_func=4.5,


        plotter.plot_PVproduction_line(
            # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
            csv_file='plot_agg_line_PVproduction___export_plot_data___17scen.csv',
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                # 'pvalloc_29nbfs_LRG2_max_sAs2p0',
                # 'pvalloc_29nbfs_LRG2_max_sAs4p0',
                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                
                # 'pvalloc_29nbfs_LRG2_max_sBs0p4',
                # 'pvalloc_29nbfs_LRG2_max_sBs0p6',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                
                # 'pvalloc_29nbfs_LRG2_max_sCs2p4',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                'pvalloc_29nbfs_LRG2_max_sCs6p8',
                
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10,],
            export_name='line_PVproduction_buABC_feedin',
            y_col='feedin_atnode_taken_kW',
            y_label='Production',
        )




print('\n*********************\n******** end ********\n*********************\n\n')







