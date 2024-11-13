# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024

# **********************************************************************************************************************************************************
# BURN LOG: Errors / Sanity Checks /  Faults          ¯\_(ツ)_/¯
# **********************************************************************************************************************************************************

# TESTING QUEUE: 
  # Check after preprep was rerun
# TODO: pred_inst_df shows instTF as false, which should be TRUE and also no pvsource



# A ====================

    # TODO: Visualize all buildings with mulitple PV installations => is problem solved with buff002 prepdata? 
    #       OTHER SOLUTION: After pv inst and egids are mapped, subselect all egids with multiple inst and map only those inst to the closest egid coordinates



    # TODO: Finalize ACCURACY:
        # OK: Do I have more or less all the houses and their roof partitions? 
            # OK > print numbers to summary log file
            # OK: > add map plot for ommitted EGIDS 
            # OK: > add charachteristics of the omitted EGIDS
        # OK: Each PV installation only fits 1 house
            # OK > print numbers to summary log file 

        # TODO: Add CKW Loadcurve from Hector
        # BOOKMARK!!

        # TODO: Select 1 house, analyse
            # > electricity prices / feedin tariffs / onetime subsidy (??)
            # > load curve
            # > production capacity, makes sense?
            # > feedin makes sense?
            # > self consumption
            # > cost of installation assumption
            # > NPV calc makes sense? 
    # ----------

    # TODO: Find a way to estimate where the existing PV installations are placed on the roof partitions. 


    # TODO: SCICORE try 1 really long run for kt:13 in scicore

        # TODO: find all instances where pvalloc script accesses data outside the output folder. Needed for scicore run
        # TODO: load a solid input data set to the cluster
        # TODO: first estimate how long it takes (how many partitions of EGIDS are stored?, what is the size of a partition?).
    # ----------

    # TODO: NPV calc is not correct, way too high returns, mean around 0.5 mio. CHF 
    # TODO: Check Loop for monthly assignments, December is always missing! => probably, december buffers out all the overshoots during the other months. different approach needed to set construction capacity constraint
    # TODO: export a map showing all the grid tarifs and DSO names per BFS


    # TODO: Visualize / export buildings, that are excluded because of too many partitions
    #       > solkat of both subsets appear to be identical (also solkats with 2-3 partitions in solkat_n_above_partitions set)



    # TODO: EGID in solkat is a faulty data point - at least for the small developing data set (nrows = ca. 1000 - 5000). 
    #       That's why a second attempt was started to aggregate roof partitions on the sbuuid and map those then to the EGID numbers. 
    #       Problem there - the amount of missmatches was strikingly large (+- 70 vs 30%) that SB_UUID aggregation does not seem to 
    #       be warrented. Also  in developing Map_solkatdfuid_egid, makes small overcounts (1 partition is attributed to two houses). 
    #       --> Check the BSBL Case thoroughly for the EGID mapping if not multiple EGIDS are attributed to one "house" (building) which then 
    #           would mean double counting
    # ----------

    # TODO: There is an issue with buildings that have a HUGE number of partitions. 
    #   > make either a function exporting all solkat with more than x rows for 1 egid
    #   > OR make a function after topo_df generation that matches the topo df to the geo data and exports it to a shape file
    #   > PLUS attach also exisiting pv installations to the topoology! so it is possible to see in topo_df which buildings have a pv installation already
    # ----------

    # TODO: some large number of EGID has multiple pv-installations attributed to them
    #   > maye use smaller buffer for gwr_gdf and pv_gdf intersection. 





# B ====================
    # TODO: Move gridprem initialization to the part of code where all TS data is imported in pvallocatio_MASTER
    # TODO: inst_power in allocation algorithm NOT weighted by angle tilt efficiency; But in updating of grid premium
            # which one makes more sense?

    # TODO: Check EGIDs in solkat for BSBL for not to have double counting.

    # ok - some egid in gwr.parquet are double up to 10 times in the column, check why!
    #       ->> because merging GWR with dwelling data, means that a row is no longer just a building but also sometimes a dwelling => multiple dwellings in one building

    # - use SolarGIS instead of meteoblue data? => not suitable for my needs



# SOLVED ====================
if True:
    # OK -  gridprem_ts not adjusting properly to grid_tiers > solved: reversed the order of conditions, so that the highest tier is checked first
    # OK adjust zoom for map plots, so that zooming in enhances the main range of the area of interest, not the margin
    # OK Get Electricity prices using another package for SPARQL APIS than the one from Elcom// from local CSV / XLSX, not through API!! => not possible
    # OK - Assign nodes better, all nodes only to EGIDs that are in sample, also solkat
    # OK - Visualize the NPV distribution 
    # OK - plot gridtiers stepf function 
    # OK gridprem update disregards selfconsumption rate; calculates 100% feedin, which would be possible. 
    # OK Define a KPI Metric (production loss ? ) and export it in the runs; idea 1 line graph HOY, 1 agg line graph totla kwH produciton loss per month / iteration

    print('')


# **********************************************************************************************************************************************************
# CODE STRUCTURE AND next steps 
# **********************************************************************************************************************************************************

# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------
if True: 
    # SETUP --------
    

    # IMPORT API DATA --------
    # OK B - Get ENTSOE API to work properly

    # still to import: 
    # OK A - TS for electricity demand
    #    > import and transformation to df from dict even sensible? => No, i just retransfrom the df to a dict again. keep it simple :)
    #    ok >> group agg_demand_df by upper and lower 50 percentilie for with/out HP and with/out PV. 
    #    ok >> then export this demand series and 
    #    ok >> attach type to EGID of GWR

    # (OK) A - TS for meteo sunshine => these two TS should ideally come from the same source
    #      > Not from the same source as demand TS -> ok, doesn't matter according also to Frank
    #      > For now just imported the shortwave radiation data (visible sunlight, amongst other columns) for Basel SBB (?)
    #      > only. Will add more weather stations later

    # (OK): Z - remove heat demand related code (not relevant if I have demand TS)


    # IMPORT LOCAL DATA + SPATIAL MAPPINGS --------
    # OK B - remove unneeded mapping functions (e.g. create_spatial_mappings, solkat_spatial_toparquet, etc.).
    # TODO: IMPORTANT: EGID in solkat is a faulty data point - at least for the small developing data set (nrows = ca. 1000 - 5000). 
    #       That's why a second attempt was started to aggregate roof partitions on the sbuuid and map those then to the EGID numbers. 
    #       Problem there - the amount of missmatches was strikingly large (+- 70 vs 30%) that SB_UUID aggregation does not seem to 
    #       be warrented. Also  in developing Map_solkatdfuid_egid, makes small overcounts (1 partition is attributed to two houses). 
    # --------> Check the BSBL Case thoroughly for the EGID mapping if not multiple EGIDS are attributed to one "house" (building) which then 
    #           would mean double counting


    # EXTEND WITH TIME FIXED DATA --------
    # OK A - remove unnecessary CUMSUM function for PV installation cost, find a way to export and store cost function params
    print('')

    # OK: A - make a summary log file, detailing how many buildings are in the data set, 
    #           how many of where dropped (why) and the same with partitions and pv inst.

# -----------------------------------------------------------------------------
# pv_allocation_MASTER.py 
# -----------------------------------------------------------------------------

if True:
    # OK - func to import and transform all prepreped data
    #   > return all relevant dfs
    # OK - func to create pv topo dict
    #   OK A - add Ausrichtung and Neigung to the topo for later uses
    #   OK A - find a functional way to adjust production to Ausrichtung (Neigung)
    #   OK A - add a fictive node to kantons 11, 12, 13 egids (just a plain csv, then merge it with empty gridprem_ts) 
    #   > return dict
    # OK - func to attach all roof partition combos to dict
    #   > return dict
    # OK - filter all input data again by alloc settings such that the allocation algorithm is applicable also on larger preped data sets

    #===============================
    # - There is an issue with buildings that have a HUGE number of partitions. 
    #  > make either a function exporting all solkat with more than x rows for 1 egid
    #  > OR make a function after topo_df generation that matches the topo df to the geo data and exports it to a shape file
    #  > PLUS attach also exisiting pv installations to the topoology! so it is possible to see in topo_df which buildings have a pv installation already
    #===============================

    # OK: A - define Construction capacity per month
    # OK: A - calculate Economics for TOPO_DF and store the partitioned DFs
    print('')

# TODO: A - add a part in calc_economic_factors that distributes existing installations to roof partitions, relative to the partition/total_roof_area ratio

# TODO: A - adjust sanity_check part:
#   > run the allocation (update gridprem_ts and update npv_df for at least 2-3 iterations)
#   > export the sanity check summary by EGID



# -----------------------------------------------------------------------------
# pv_alloc_MCalgorithm_MASTER.py 
# -----------------------------------------------------------------------------

# TODO: rename "pv_allocation_MASTER.py" to "pv_alloc_initialization_MASTER.py"
# TODO: rename "name_dir_export" in pvalloc_settings to "name_dir_pvalloc_init_export"
# TODO: create a new setting in settingsfile :          "name_dir_pvalloc_MCalgo_export"


# SETUP
# TODO: A - create a directory to store all the monte carlo iterations

# LOOP for MONTE CARLO
#   > TODO: create dir for this single iteration
#   > TODO: copy all the data objects that are needed "fresh" for each iteration. oterhwise iter2 
#     will be influenced by iter1 (will start where iter1 ended. 

#        - LOOP for MONTH
#          - update npv for all non pv buildings
#            OK A - get all topo_time_subdf from many seperated sub_topo_dfs.parquet by EGID
#            > return npv_df
#       
#          - func to update feedin premium
#            OK A - get all installations from topo.dict
#            OK A - calculate production HOYs and find where voltage level meets tier
#            OK A - update gridprem_ts
#            > return premium TS
#       
#          - pick installation
#            OK A - create selection mechanism NPV based as function
#            OK A - create random selection mechanism as function (for comparision)
#       
#          - add picked installation to topo.dict 
#       
#          - export built installations for later visualizations

#   - #TODO:OK A - select only the relevant output files to store in MC dir


# -----------------------------------------------------------------------------
# visualizations_MASTER.py 
# -----------------------------------------------------------------------------

# Create plot functions for the following:

# - individual scenarios
#   OK A - Amount of installed capacity per node (avg incl std bands)
#   OK A - Amount of installed capacity per bfs (avg incl std bands)
#   OK ? - Map of covered regions (bfs) with installed capacity
#   OK: ? - Avg building characteristics (amount of houses in each class etc., with x roof partitions etc.)


# - aggregated scenarios
#   OK A - Avg Amount of installed capacity in months (avg incl std bands)
#   OK A - Avg Production HOY per scenario (avg incl std bands)
#   TODO: A - Final avg gridprem ts per scenario (avg incl std bands)

#   OK A - Map of covered regions (bfs) with installed capacity
#   OK A - Map of covered regions (bfs) with sum production (maybe incl solkat)



#   - repick building for PV
# -----------------------------------------------------------------------------
# data / model extensions 
# -----------------------------------------------------------------------------
# MA Lupien, Thormeyer et al 2020; also use protected / heritage zones to rule out PVs in that area
# MA Lupien, EV stations per Gemeinde => possible to map stations to DSO nodes?
# MA Lupien, "Exploitable solar resource" per gmeinde?
