# OptimalPV 

## Working Paper:
**Author**: raul.hochuli@unibas.ch \
**Latest update**: Working Paper: Jan 2026, README: Jan 2026 \
Find the current version of [OptiamlPV working paper](docs/OptimalPV_RH_v1_260107_WorkingPaper.pdf), my signle author paper for my PhD, here for more elabortion on the scientific contribution. 

> **Abstract**: Due to the global push for distributed photovoltaic (PV) systems, grid capacity will be a limiting factor for further expansion in the near future. Additional installations can cause a negative externality by hindering other production in the same grid circuit. 
This calls for a detailed modeling approach that captures the negative impact of a PV installation on neighbor's production capability. 
Using spatial data at the individual house level for ca. 40 thousand single-family houses, I propose a stochastic Monte Carlo method to model theoretical PV expansion pathways until 2050 from a social planer's perspective considering optimal grid ussage and a economic rational individual's perspective. 
I use a local grid operators data on low distribution house-grid connections and transformer capacities. This is the cornerstone for further research on a dynamic feed-in permit that is designed to mitigate grid congestion without requiring intense regulator monitoring.  


## Main Execution Scripts
- The [main_INIT_array.py](main_INIT_array.py) file serves as the primary entry point for running the PV allocation model, calling `MAIN_pvallocation.py` to execute model simulations that determine residential  photovoltaic installation pathways across municipalities under various scenarios. 
- The [main_visualization.py](main_visualization.py) file handles post-processing and visualization tasks by calling `MAIN_visualization` to generate comprehensive plots, maps, and analytical outputs from the allocation results. 
All script files (`.py`, `.sh`, `.cmd`) located in the main directory of this repository are specifically designed for parallel model execution on the university's High-Performance Computing (HPC) cluster, enabling efficient processing of computationally intensive Monte Carlo iterations across multiple nodes.

## Scenario Settings & Assumptions
[MAIN_pvallocation.py](src/[MAIN_pvallocation.py) is structured using two classes, where the model execution is run in `PVAllocScenario`, serving all scenario settings from `PVAllocScenario_Settings`. This structure allows for flexible configuration of model parameters such as municipality selection, time horizons, technology specifications, and allocation algorithms while maintaining a clean separation between scenario configuration and execution logic.

Before running the PV allocation model, [MAIN_data_aggregation.py](src/MAIN_data_aggregation.py) must be executed first to transform all relevant data sources into a usable format. This preprocessing step integrates heterogeneous datasets including building registry data (GWR), grid topology information from the local distribution system operator, meteorological data, and PV installation records. The aggregation process standardizes coordinate systems, matches buildings to grid nodes, calculates solar radiation potentials, and prepares all necessary input files that the allocation model requires for its Monte Carlo simulations.

[calibration_class.py](src/calibration_class.py) also needs to be executed first before any PV allocation models can be run. This file calibrates and stores a selection of prediction estimators used to determine the PV capacity that is installed on future house in [MAIN_pvallocation.py](src/[MAIN_pvallocation.py), determined by the setting `ALGOspec_pvinst_size_calculation` and `ALGOspec_calib_estim_mod_name_pkl`.
<!-- 
## Code Structure and Process
For a better understanding, review the visualization of the model structure in the **[readme_OptimalPV_graph.png](docs/readme_OptimalPV_graph.png)**. The two files:
- [execution_scenarios.py](execution_scenarios.py) (defining all scenario settings (just deviations from default settings) for data aggregation and pv allocation model runs)
- [execution_RUNFILE.py](execution_RUNFILE.py) (the main file, running (selected parts of the model) with selected scenarios and visualization settings)

- **MASTER files:** The MASTER functions use docstrings to explain in a few steps what the called subfiles and functions are doing. Because they represent a larger process with many steps, MASTER functions do not return any objects but store various datafiles in an interm folder directory, which is then again accessed by the next MASTER function. This allows for development of only small specific parts in the entire model and later parallelization of the Monte Carlo iterations. 
- **Setting dictionaries:** Each MASTER file has a corresponding settings directory containing relevant information for the specific part of the model. These setting directories have default values, stored in the *default_settings.py* (e.g. [default_settings.py](pv_allocation/default_settings.py)) file of every code subfolder. For increased readabiligy, the run scenarios for the data aggregation and pv allocation step are defined in *execution_scenarios.py*. There only the changes to the default values are written down to increase visibility. The remaining default values for each scenario are later added with a function in [execution_RUNFILE.py](execution_RUNFILE.py).



For a detailed account of what the individual subfiles and functions are doing, read the [OptimalPV_RH_ModelDescription.pdf](docs/OptimalPV_RH_ModelDescription.pdf) which tries to stay up to date with all functions and their purpose. This model description will be transferred step by step to an individual README file in each subfolder. 
 -->

