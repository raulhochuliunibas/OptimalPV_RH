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
        export_dir_list = ['paper_publication', 'WorkingPaper_v3_OptimalPV_RH', 'figures'],
    )
    plotter.line_opacity    = 0.65
    plotter.plot_width      = 4
    plotter.plot_height     = 5.3
    plotter.plot_dpi        = 300
    plotter.show_plt_TF     = False

    plotter.scen_default_color_map = {
        'pvalloc_LRG4_max_fixpv2023':          (200, 50, 50),      # red (keep)
        # 'pvalloc_26nbfs_LRG_max_1hll':      (200, 140, 40),  # orange (instead of yellow)
        # 'pvalloc_LRG4_max_fixpv2023_As6p0':      (                               
        # A = orange, B = green, C = blue; hoypv2023 variants share the tone of their fixpv2023 counterpart
        'pvalloc_LRG4_max_fixpv2023_As8p0':     (230, 130, 30),  # orange
        'pvalloc_LRG4_max_fixpv2023_Bs0p8':     (60, 160, 90),   # green (less neon)
        'pvalloc_LRG4_max_fixpv2023_Cs6p6':     (40, 90, 190),   # blue
        'pvalloc_LRG4_max_fixpv2023_Cs6p8':     (90, 140, 220),  # light blue
        'pvalloc_LRG4_max_fixpv2023_Cs8p6':     (20, 60, 140),   # dark blue
        'pvalloc_LRG4_max_fixpv2023_Cs8p8':     (60, 170, 210),  # cyan-blue

        'pvalloc_LRG4_max_hoypv2023':           (200, 50, 50),   # red, same as BAU (dashed)
        'pvalloc_LRG4_max_hoypv2023_As8p0':     (230, 130, 30),  # orange
        'pvalloc_LRG4_max_hoypv2023_Bs0p8':     (60, 160, 90),   # green
        'pvalloc_LRG4_max_hoypv2023_Cs6p6':     (40, 90, 190),   # blue
        'pvalloc_LRG4_max_hoypv2023_Cs6p8':     (90, 140, 220),  # light blue
        'pvalloc_LRG4_max_hoypv2023_Cs8p6':     (20, 60, 140),   # dark blue
        'pvalloc_LRG4_max_hoypv2023_Cs8p8':     (60, 170, 210),  # cyan-blue
        # 'pvalloc_26nbfs_LRG_max_histconstr_e40pv6':         (140, 60, 180),          
        # 'pvalloc_26nbfs_LRG_max_nokev':                     (200, 140, 40),          
        # # 'pvalloc_26nbfs_LRG_max_pksh07',                              
        # 'pvalloc_26nbfs_LRG_max_pksh09',                              
        # 'pvalloc_26nbfs_LRG_rnd',                                     
        'pvalloc_LRG4_max_fixpv2023_gridoptim':    (180, 140, 190),    # lilac


        'pvalloc_DEV_May26_max_kev':  (200, 50, 50),      # red (keep)
        'pvalloc_DEV_Oct2_26_max_kev':  (200, 50, 50),      # red (keep)
        'pvalloc_DEV_May26_max_As8p0': (230, 130, 30),  # orange
        'pvalloc_DEV_Oct2_26_max_As8p0': (230, 130, 30),  # orange
        'pvalloc_DEV_May26_max_Bs0p8': (60, 160, 90),   # green (less neon)
        'pvalloc_DEV_Oct2_26_max_Bs0p8': (60, 160, 90),   # green (less neon)
        'pvalloc_DEV_May26_max_Cs8p8': (60, 170, 210),  # cyan-blue
        'pvalloc_DEV_Oct2_26_max_Cs8p8': (60, 170, 210),  # cyan-blue
        'pvalloc_DEV_May26_gridoptim' : (180, 140, 190),    # lilac
        'pvalloc_DEV_Oct2_26_gridoptim' : (180, 140, 190),    # lilac

    }
    plotter.simple_scen_name_mapping = {
        'pvalloc_LRG4_max_fixpv2023':                     'BAU (default)', 
        # 'pvalloc_26nbfs_LRG_max_1hll':                'DSO node closing', 
        # 'pvalloc_LRG4_max_fixpv2023_As8p0':               'Sub. A (8000, 0)', 
        # 'pvalloc_LRG4_max_fixpv2023_Bs0p8':               'Sub. B (0, 8000)', 
        # 'pvalloc_LRG4_max_fixpv2023_Cs6p6':               'Sub. C (6000, 6000)', 
        # 'pvalloc_LRG4_max_fixpv2023_Cs6p8':               'Sub. C (6000, 8000)', 
        # 'pvalloc_LRG4_max_hoypv2023':                'HOY pElec.2023', 
        # 'pvalloc_26nbfs_LRG_max_histconstr_e40pv6':   'Elec40_pv6',
        # 'pvalloc_26nbfs_LRG_max_nokev':               'No Subsidy',
    }
    plotter.scen_default_linedash_marker_map = {
        'pvalloc_LRG4_max_fixpv2023':                ('solid', 'o'), 
        'pvalloc_LRG4_max_fixpv2023_As8p0':          ('solid', 'o'), 
        'pvalloc_LRG4_max_fixpv2023_Bs0p8':          ('solid', 'o'), 
        'pvalloc_LRG4_max_fixpv2023_Cs6p6':          ('solid', 'o'),
        'pvalloc_LRG4_max_fixpv2023_Cs6p8':          ('solid', 'o'),
        'pvalloc_LRG4_max_fixpv2023_Cs8p6':          ('solid', 'o'),
        'pvalloc_LRG4_max_fixpv2023_Cs8p8':          ('solid', 'o'),
        'pvalloc_LRG4_max_hoypv2023':                ('dashed', 'o'),
        'pvalloc_LRG4_max_hoypv2023_As8p0':          ('dashed', 'o'),
        'pvalloc_LRG4_max_hoypv2023_Bs0p8':          ('dashed', 'o'),
        'pvalloc_LRG4_max_hoypv2023_Cs6p6':          ('dashed', 'o'),
        'pvalloc_LRG4_max_hoypv2023_Cs6p8':          ('dashed', 'o'),
        'pvalloc_LRG4_max_hoypv2023_Cs8p6':          ('dashed', 'o'),
        'pvalloc_LRG4_max_hoypv2023_Cs8p8':          ('dashed', 'o'),

        'pvalloc_DEV_May26_max_kev':                  ('solid', 'o'),
        'pvalloc_DEV_May26_max_As8p0':                ('solid', 'o'),
        'pvalloc_DEV_May26_max_Bs0p8':                ('solid', 'o'),
        'pvalloc_DEV_May26_max_Cs8p8':                ('solid', 'o'),
        'pvalloc_DEV_May26_gridoptim' :               ('solid', 'o'),
        'pvalloc_DEV_Oct2_26_max_kev':                ('dashed', 'o'),
        'pvalloc_DEV_Oct2_26_max_As8p0':              ('dashed', 'o'),
        'pvalloc_DEV_Oct2_26_max_Bs0p8':              ('dashed', 'o'),
        'pvalloc_DEV_Oct2_26_max_Cs8p8':              ('dashed', 'o'),
        'pvalloc_DEV_Oct2_26_gridoptim' :             ('dashed', 'o'),
   
    }



    # plotter.copy_standalone_graphs_to_presentation_dir()

    # SCEN - Excess Feedin + PV Production tables ======================================================================
    PVprod_csv_file = 'plot_agg_line_PVproduction___export_plot_data___15scen.csv'
    PVprod_scen_list = [
        'pvalloc_DEV_May26_gridoptim', 
        'pvalloc_DEV_May26_max_As8p0', 
        'pvalloc_DEV_May26_max_Bs0p8', 
        'pvalloc_DEV_May26_max_Cs8p8', 
        'pvalloc_DEV_May26_max_kev', 
        ]
    PVprod_scen_list2 = [
        'pvalloc_DEV_Oct2_26_gridoptim', 
        'pvalloc_DEV_Oct2_26_max_As8p0', 
        'pvalloc_DEV_Oct2_26_max_Bs0p8', 
        'pvalloc_DEV_Oct2_26_max_Cs8p8', 
        'pvalloc_DEV_Oct2_26_max_kev', 
    ]


    def plot_gridnode_HOY_wrapper(plotter , plot_width = 8, plot_height = 4, 
                                  n_iter_range_list_func = [  
                                      1, 2, 3, 4, 5,
                                      6, 7, 8,
                                        9,
                                        10, 11, 
                                        12, 13, 14, 15, 16, 17, 18, 19, 20,],
                                  ):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height

        # Exceed Feedin 
        plotter_func.plot_PVproduction_line(
            csv_file=PVprod_csv_file, 
            # csv_file= 'plot_agg_line_PVproduction___export_plot_data___25scen.csv',
            scen_incl_list= PVprod_scen_list, 
            n_iter_range_list= n_iter_range_list_func,
            export_name='line_PVproduction_bau_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6, 
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        
        plotter_func.plot_PVproduction_line(
            csv_file=PVprod_csv_file, 
            # csv_file= 'plot_agg_line_PVproduction___export_plot_data___25scen.csv',
            scen_incl_list= PVprod_scen_list2, 
            n_iter_range_list= n_iter_range_list_func,
            export_name='line_PVproduction_bau_excfeedin2',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6, 
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )

        # Taken Production
        plotter_func.plot_PVproduction_line(
            csv_file=PVprod_csv_file, 
            # csv_file= 'plot_agg_line_PVproduction___export_plot_data___25scen.csv',
            scen_incl_list= PVprod_scen_list, 
            n_iter_range_list=n_iter_range_list_func,
            export_name='line_PVproduction_bau_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )
        plotter_func.plot_PVproduction_line(
            csv_file=PVprod_csv_file, 
            # csv_file= 'plot_agg_line_PVproduction___export_plot_data___25scen.csv',
            scen_incl_list= PVprod_scen_list2, 
            n_iter_range_list=n_iter_range_list_func,
            export_name='line_PVproduction_bau_feedin2',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )
    
    plot_gridnode_HOY_wrapper(plotter, plot_width= 8,  plot_height=8 )


    print(f'\n*************************\n**** end plots {__file__} ****\n*************************\n')


if False:


    print('-')
    # AGG CSV - based plots ======================================================================

    def plot_PVprod_SUBS_OPT_wrapper(plotter_func, plot_width = 8, plot_height = 4):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height


        print(' = SUBs case =====================')
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sAs8p0',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bauABC_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6 ,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sAs8p0',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bauABC_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',

        )


        print(' = GridOptim case =====================')
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                # 'pvalloc_LRG3_max_sCs4p6',
                'pvalloc_LRG3_max_gridoptim',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_gridoptim_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                # 'pvalloc_LRG3_max_sCs4p6',
                'pvalloc_LRG3_max_gridoptim',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_gridoptim_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )

    plot_PVprod_SUBS_OPT_wrapper(plotter, plot_width= 4.8, plot_height= 5.2)




    # SCEN - individual tables + data ======================================================================
    if False: 
        
        # plotter.plot_width      = 6.3
        # plotter.plot_height     = 3.6
        plotter.plot_width      = 4.1
        plotter.plot_height     = 2.4

        plotter.plot_EGID_pvprod_demand_HOY(export_name = 'plot_EGID_HOY_pvprod',
                                            plot_cols_incl_list= ['pvprod_kW'],)
        plotter.plot_EGID_pvprod_demand_HOY(export_name = 'plot_EGID_HOY_pvprod_demand',
                                            plot_cols_incl_list= ['pvprod_kW', 'demand_kW'],)
        plotter.plot_EGID_pvprod_demand_HOY(export_name = 'plot_EGID_HOY_pvprod_demand_feedin',
                                        plot_cols_incl_list= ['pvprod_kW', 'demand_kW', 'netfeedin_kW'],)


        # plotter.get_single_values(
        #     csv_file='plot_agg_line_PVproduction___export_plot_data___14scen.csv',
        #     scen_incl_list=[
        #     #     'pvalloc_LRG3_max_1hll',
        #     #     'pvalloc_LRG3_max_1hll_sCs4p6',
        #            'DEV_pvalloc_29nbfs_LRG_max_OLDpreprep',
        #            'DEV_pvalloc_29nbfs_LRG_max_sCs4p6_OLDpreprep',
        #         ],
        #         n_iter_range_list=[20,],

        # )
        

        plotter.plot_width      = 4
        plotter.plot_height     = 3
        plotter.NPVhist( scen = 'pvalloc_LRG3_max',             export_name='NPVhist_LRG3_max', title= 'NPV Distribution 30Rp/kWh',  npv_hist_xrange=(-15e3, 45e3), )

        # plotter.plot_width      = 2.8
        # plotter.plot_height     = 5.0
        plotter.plot_width      = 4.0
        plotter.plot_height     = 4.0
        plotter.NPVhist( scen = 'pvalloc_LRG3_max',             export_name='NPVhist_LRG3_max_31Rp', title= 'NPV Dist. (default 31.9 Rp/kWh)',  npv_hist_xrange=(-15e3, 45e3), )
        plotter.NPVhist( scen = 'pvalloc_LRG3_max_elecpri15RP', export_name='NPVhist_LRG3_max_15Rp',   title= 'NPV Dist. (15Rp/kWh)',  npv_hist_xrange=(-15e3, 45e3), )
        plotter.NPVhist( scen = 'pvalloc_LRG3_max_elecpri60RP', export_name='NPVhist_LRG3_max_60Rp',   title= 'NPV Dist. (60Rp/kWh)',  npv_hist_xrange=(-15e3, 12e4), )
        
        plotter.DataSampleSummary()

        # plotter.Loss_Subscost_Summary(
        #     scen_list = ['pvalloc_LRG3_max', 
        #                  'pvalloc_LRG3_max_sCs6p8',
        #     ], 
        #     n_iter_list = [4, 6, 8, ],
        # )


    print('-')
    # SCEN - individual plots  ======================================================================
    PVprod_csv_file = 'plot_agg_line_PVproduction___export_plot_data___26scen.csv'

    def plot_gridnode_HOY_wrapper(plotter , plot_width = 8, plot_height = 4):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height
        rgb_red     = (200, 50, 50)
        rgb_green   = (60, 180, 60)
        rgb_yellow = (200, 140, 40)
        rgb_grey  = (150, 150, 150)
        rgb_nokev    = rgb_grey

        # HOY
        plotter_func.scen_default_color_map = {
            'pvalloc_LRG3_max':          rgb_red,      # red (keep)
            'pvalloc_LRG3_max_nokev':    rgb_nokev,      # red (keep)
        }
        plotter_func.simple_scen_name_mapping = {
            'pvalloc_LRG3_max': 'Business-as-usual',
            'pvalloc_LRG3_max_nokev': 'No Subsidy',
        }
        plotter_func.plot_gridnode_HOY( 
            scen                = 'pvalloc_LRG3_max' ,
            hours_incl_list     = list(range(4920 + 3*24, 4920 + 5*24)),
            iter                = 8,
            # gridnode            = '754',
            below_threshold_rgb = rgb_green,
            above_threshold_rgb = rgb_red,
        )

        # Exceed Feedin 
        plotter_func.plot_PVproduction_line(
            csv_file=PVprod_csv_file, 
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_nokev',
                            ],            
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bau_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6, 
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )

        # Taken Production  
        plotter_func.scen_default_color_map = {
            'pvalloc_LRG3_max':          rgb_green,     
            'pvalloc_LRG3_max_nokev':    rgb_nokev,      
        }
        plotter_func.plot_PVproduction_line(
            csv_file=PVprod_csv_file, 
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_nokev',
                            ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bau_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )
    
    # plot_gridnode_HOY_wrapper(plotter, plot_width= 3.2, plot_height= 5.2)


    print('-')
    # AGG CSV - based plots ======================================================================

    def plot_PVprod_SUBS_OPT_wrapper(plotter_func, plot_width = 8, plot_height = 4):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height


        print(' = SUBs case =====================')
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sAs8p0',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bauABC_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6 ,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sAs8p0',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_bauABC_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',

        )


        print(' = GridOptim case =====================')
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                # 'pvalloc_LRG3_max_sCs4p6',
                'pvalloc_LRG3_max_gridoptim',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_gridoptim_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_sBs0p8',
                'pvalloc_LRG3_max_sCs6p8',
                # 'pvalloc_LRG3_max_sCs4p6',
                'pvalloc_LRG3_max_gridoptim',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_gridoptim_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )

    # plot_PVprod_SUBS_OPT_wrapper(plotter, plot_width= 4.8, plot_height= 5.2)

    PVprod_csv_file = 'plot_agg_line_PVproduction___export_plot_data___26scen.csv'
    print('-')
    def plot_PVprod_DSO_wrapper(plotter_func, plot_width = 8, plot_height = 4):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height

        print(' = DSO react case =====================')
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_1hll',
                'pvalloc_LRG3_max_pksh',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_DSOreact_excfeedin',
            y_col='feedin_atnode_loss_kW',
            y_scaling= 1e6,
            y_label='Agg. Excess Feed-in (GWh)',
            title='Agg. Exc Feedin',
        )
        plotter_func.plot_PVproduction_line(
            csv_file= PVprod_csv_file,
            scen_incl_list=[
                'pvalloc_LRG3_max',
                'pvalloc_LRG3_max_1hll',
                'pvalloc_LRG3_max_pksh',
                ],
            n_iter_range_list=[5,  6, 7, 8, 9, 10, 11, 12, ],
            export_name='line_PVproduction_DSOreact_feedin',
            y_col='feedin_atnode_taken_kW',
            y_scaling= 1e6,    
            y_label='Aggregated Feed-in (GWh)',
            title='Agg. Feed-in',
        )

    # plot_PVprod_DSO_wrapper(plotter, plot_width= 4.8, plot_height= 5.2)


    def plot_worstweek_wrapper(plotter_func, plot_width = 8, plot_height = 4):
        
        plotter_func = plotter.copy()
        plotter_func.plot_width  = plot_width
        plotter_func.plot_height = plot_height

        print(' = Worst week case =====================')
        # plotter_func.worstnode_worstweek(
        #     scen = 'pvalloc_LRG3_max',
        #     excess_feedin_pegid = False,
        #     export_name = 'worstnode_worstweek_total',
        #     title = 'Excess Feedin Total ("Worst Week")',
        #     plot_width_func = plot_width,
        #     plot_height_func = plot_height,
        # )
        plotter_func.worstnode_worstweek(
            scen = 'pvalloc_LRG3_max',
            excess_feedin_pegid = True,
            export_name = 'worstnode_worstweek_pegid',
            title = 'Avg. Exc. Feedin pHouse (Red: Peak Day)',
            plot_width_func = plot_width,
            plot_height_func = plot_height,
        )

        plotter_func.hist_avgloss_pEGID(
            scen = 'pvalloc_LRG3_max',
            title = 'Avg. Excess Feedin pHouse - All Nodes',
            export_name = 'hist_avgloss_pEGID',
            x_label = 'Excess Feed-in per House in 24h (kWh)',
            y_label = 'Frequency (n Nodes)',
            plot_width_func = plot_width,
            plot_height_func = plot_height,
        )

    # plot_worstweek_wrapper(plotter, plot_width= 4.8, plot_height= 5.2)
    # plot_worstweek_wrapper(plotter, plot_width = 5.0, plot_height = 2.8)




    # Inst Charact Comparison plots ===========================================================

    if True: 
        contcharacht_csv_file = 'plot_agg_hist_contcharact_newinst___export_plot_data___23scen.csv'
        # bu_contcharact_height   = 3 
        # bu_contcharact_width    = 8.5
        bu_contcharact_height   = 1.9
        bu_contcharact_width    = 4.8

        catgcharacht_csv_file = 'plot_agg_bar_catgcharact_newinst___export_plot_data___23scen.csv'
        # bu_catgcharact_height   = 2.85
        # bu_catgcharact_width    = 2.4
        bu_catgcharact_height   = 1.9
        bu_catgcharact_width    = 2.4

        print(' = Charact comparison =====================')
        # plotter = static_plotter_class()
        # plotter.plot_ind_hist_contcharact_newinst(
        #     csv_file=contcharacht_csv_file,
        #     scen_incl_list=['pvalloc_LRG3_max',
        #                     # 'pvalloc_LRG3_max_sAs8p0',
        #                     'pvalloc_LRG3_max_sBs0p8',
        #                     # 'pvalloc_LRG3_max_sCs6p8',
        #                     ],
        #     iter_incl_list=[1,
        #                     4 
        #                     # 3, 5
        #                     ],
        #     x_col_incl_list=['FLAECHE', 'GAREA'],
        #     export_name='hist_contcharact_newinst_bu',
        #     plot_height_func = bu_contcharact_height,
        #     plot_width_func  = bu_contcharact_width,
        # )
        # plotter.plot_ind_line_catgcharact_newinst(
        #     csv_file=catgcharacht_csv_file,
        #     scen_incl_list=['pvalloc_LRG3_max',
        #                     # 'pvalloc_LRG3_max_sAs8p0',
        #                     'pvalloc_LRG3_max_sBs0p8',
        #                     # 'pvalloc_LRG3_max_sCs6p8',
        #                     ],
        #     iter_incl_list=[
        #         1, 2, 3, 4, 5, 6, 
        #         # 7, 8, 9, 10, 11, 12, 13, 14, 15,
        #                       ],
        #     x_col_incl_dict={
        #         'GKLAS': {
        #             # 'single-family':['1110',], 
        #             # 'multi-family':['1121', '1122', ]
        #             '1 apart.':['1110',],
        #             '2 apart.':['1121', ],
        #             '3+ apart.':['1122', ]
        #                 },
        #         'are_typ': {
        #             'rural':['Rural',],
        #             'suburban':['Suburban',],
        #             'urban':['Urban',]
        #                     },
        #         'heatpump_TF': {
        #             'HP':['heatpump',],
        #             'no HP':['no_heatpump',]
        #                     }, 
        #         'filter_tag': {
        #             'east-west': ['eastwest_80pr', 'eastwest_70pr'],
        #             'south': ['south_50pr', 'south_40pr'],
        #         },
        #     },
        #     export_name='line_catgcharact_newinst_bu',
        #     plot_height_func = bu_catgcharact_height,
        #     plot_width_func  = bu_catgcharact_width,
        # )


