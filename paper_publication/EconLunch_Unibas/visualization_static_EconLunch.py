import sys
import os


# ensure repo root is on sys.path so `src` package can be imported when
# running this script from the paper_publication/EconLunch_Unibas folder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.visualization_static.visualization_static import static_plotter_class

if __name__ == "__main__":

    plotter = static_plotter_class(
        export_dir_list = ['paper_publication', 'EconLunch_Unibas', 'figures'],
    )
    plotter.line_opacity    = 0.65
    plotter.plot_width      = 4
    plotter.plot_height     = 5.3
    plotter.plot_dpi        = 300
    plotter.show_plt_TF     = False

    plotter.scen_default_color_map = {
        'pvalloc_29nbfs_LRG2_max':          (200, 50, 50),      # red (keep)
    
        'pvalloc_29nbfs_LRG2_max_epzb1':                (230, 140, 140),  # soft pastel red
        'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1':    (60, 180, 60),                # bright green           
        'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3':    (150, 220, 150),               # warm light orange (instead of yellow)

        'pvalloc_29nbfs_LRG2_max_sAs6p0':   (60, 120, 200),  # strong blue
        'pvalloc_29nbfs_LRG2_max_sBs0p8':   (60, 160, 90),   # green (less neon)
        'pvalloc_29nbfs_LRG2_max_sCs2p8':   (200, 140, 40),  # orange (instead of yellow)

        'pvalloc_29nbfs_LRG2_max_1hll':     (100, 180, 180),  # turquoise (instead of yellow)
        'pvalloc_29nbfs_LRG2_max_pksh':      (0, 80, 80),  # dark teal (instead of cyan)
        
        'pvalloc_29nbfs_LRG2_gridoptim_max':    (140, 60, 180),    # bright purple
        # 'pvalloc_29nbfs_LRG2_gridoptim_max':    (160, 120, 180),    # lavendar
        # 'pvalloc_29nbfs_LRG2_gridoptim_max':    (180, 140, 190),    # lilac
    }
    plotter.simple_scen_name_mapping = {
        'pvalloc_29nbfs_LRG2_max':              'Business-as-usual',
        'pvalloc_29nbfs_LRG2_max_sAs6p0':       'Subsidy A (6000, 0)',
        'pvalloc_29nbfs_LRG2_max_sBs0p8':       'Subsidy B (0, 8000)',
        'pvalloc_29nbfs_LRG2_max_sCs2p8':       'Subsidy C (2000, 8000)',
        'pvalloc_29nbfs_LRG2_max_1hll':         'DSO node closing',
        'pvalloc_29nbfs_LRG2_max_pksh':         'DSO peak shaving',
        'pvalloc_29nbfs_LRG2_gridoptim_max':    'Grid-optimized',
    }


    # SCEN - Individual Tables + Data ======================================================================
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

        # plotter.NPVhist_DataSampleSummary()

        # plotter.Loss_Subscost_Summary(
        #     scen_list = ['pvalloc_29nbfs_LRG2_max', 
        #                  'pvalloc_29nbfs_LRG2_max_sCs2p8',
        #     ], 
        #     n_iter_list = [4, 6, 8, ],
        # )



    # SCEN - Individual Plots plots ======================================================================
    
    def plot_gridnode_HOY_wrapper(plotter , plot_width = 8, plot_height = 4):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height
        rgb_red     = (200, 50, 50)
        rgb_green   = (60, 180, 60)

        # HOY
        plotter_func.scen_default_color_map = {
            'pvalloc_29nbfs_LRG2_max':          rgb_red,      # red (keep)
        }
        plotter_func.simple_scen_name_mapping = {
            'pvalloc_29nbfs_LRG2_max': 'Business-as-usual',
        }
        plotter_func.plot_gridnode_HOY( 
            hours_incl_list=list(range(4920 + 3*24, 4920 + 5*24)),
            below_threshold_rgb = rgb_green,
            above_threshold_rgb = rgb_red,
        )

        # Exceed Feedin 
        plotter_func.plot_PVproduction_line(
            csv_file='plot_agg_line_PVproduction___export_plot_data___21scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bau_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6, 
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )

        # Taken Production  
        plotter_func.scen_default_color_map = {
            'pvalloc_29nbfs_LRG2_max':          rgb_green,
        }
        plotter_func.plot_PVproduction_line(
            csv_file='plot_agg_line_PVproduction___export_plot_data___21scen.csv',
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bau_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )
    
    plot_gridnode_HOY_wrapper(plotter, plot_width= 3.2, plot_height= 5.3)



    # AGG CSV - based plots ======================================================================
    PVprod_csv_file = 'plot_agg_line_PVproduction___export_plot_data___21scen.csv'
    if True: 
        
        # print('- BU: plot_PVproduction_line')
        # plotter.plot_PVproduction_line(
        #     # csv_file='plot_agg_line_PVproduction___export_plot_data___1scen.csv',
        #     csv_file= PVprod_csv_file,
        #     scen_incl_list=['pvalloc_29nbfs_LRG2_max',
        #                     # 'pvalloc_29nbfs_LRG2_max_epzb1', 
        #                     'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_1',
        #                     # 'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_2',
        #                     'pvalloc_29nbfs_LRG2_max_histcnstrcapgr0_3',
        #                     ],
        #     n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12],
        #     export_name='line_PVproduction_bu_loss',
        #     y_col='feedin_atnode_loss_kW',
        #     y_scaling= 1e6, 
        #     y_label='Agg. Excess Feed-in (GWh)'
        #     title='Agg. Exc Feedin',
        # )

        

        print(' = SUBs case =====================')
        plotter.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bauABC_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6 ,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sAs6p0',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bauABC_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',

        )


        print(' = GridOptim case =====================')
        plotter.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                'pvalloc_29nbfs_LRG2_gridoptim_max',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_gridoptim_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_sBs0p8',
                'pvalloc_29nbfs_LRG2_max_sCs2p8',
                # 'pvalloc_29nbfs_LRG2_max_sCs4p6',
                'pvalloc_29nbfs_LRG2_gridoptim_max',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_gridoptim_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )


        print(' = DSO react case =====================')
        plotter.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_1hll',
                'pvalloc_29nbfs_LRG2_max_pksh',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_DSOreact_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_29nbfs_LRG2_max',
                'pvalloc_29nbfs_LRG2_max_1hll',
                'pvalloc_29nbfs_LRG2_max_pksh',
                ],
            n_iter_range_list=[4, 5, 6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_DSOreact_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )




    # Inst Charact Comparison plots ===========================================================

    if False: 
        contcharacht_csv_file = 'plot_agg_hist_contcharact_newinst___export_plot_data___18scen.csv'
        # bu_contcharact_height   = 3 
        # bu_contcharact_width    = 8.5
        bu_contcharact_height   = 2.2 
        bu_contcharact_width    = 4.8

        catgcharacht_csv_file = 'plot_agg_bar_catgcharact_newinst___export_plot_data___18scen.csv'
        # bu_catgcharact_height   = 2.85
        # bu_catgcharact_width    = 2.4
        bu_catgcharact_height   = 2.22
        bu_catgcharact_width    = 2.4

        print(' = Charact comparison =====================')
        plotter = static_plotter_class()
        plotter.plot_ind_hist_contcharact_newinst(
            csv_file=contcharacht_csv_file,
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',
                            # 'pvalloc_29nbfs_LRG2_max_sAs6p0',
                            'pvalloc_29nbfs_LRG2_max_sBs0p8',
                            # 'pvalloc_29nbfs_LRG2_max_sCs2p8',
                            ],
            iter_incl_list=[1,
                            4 
                            # 3, 5
                            ],
            x_col_incl_list=['FLAECHE', 'GAREA'],
            export_name='hist_contcharact_newinst_bu',
            plot_height_func = bu_contcharact_height,
            plot_width_func  = bu_contcharact_width,
        )
        plotter.plot_ind_line_catgcharact_newinst(
            csv_file=catgcharacht_csv_file,
            scen_incl_list=['pvalloc_29nbfs_LRG2_max',
                            # 'pvalloc_29nbfs_LRG2_max_sAs6p0',
                            'pvalloc_29nbfs_LRG2_max_sBs0p8',
                            # 'pvalloc_29nbfs_LRG2_max_sCs2p8',
                            ],
            iter_incl_list=[
                1, 2, 3, 4, 5, 6, 
                # 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              ],
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
            plot_height_func = bu_catgcharact_height,
            plot_width_func  = bu_catgcharact_width,
        )


    print('\n*********************\n******** end ********\n*********************\n\n')
