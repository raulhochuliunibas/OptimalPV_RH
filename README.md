# OptimalPV 

## Working Paper:
**Author**: raul.hochuli@unibas.ch \
**Latest Version**: *Decisions at the Grid's Edge: An Improved Spatial PV Expansion in Swiss Residences* \
Find the current version of the [OptimalPV working paper](paper_publication/WorkingPaper_v2_OptimalPV_RaulHochuli/v2_OptimalPV_WorkingPaper_compressed.pdf), my single-author paper for my PhD, here for more elaboration on the scientific contribution. \
**Latest update**:
- Working Paper:  Jan 2026 (v2, conference paper)
- README:         Nov 2026

**HPC**: Calculations were performed at sciCORE (http://scicore.unibas.ch/) scientific computing center at University of Basel.

> **Abstract**: The rapid expansion of residential photovoltaic (PV) installations is a central pillar of the transition toward a renewable energy production. However, increasing PV penetration is beginning to create congestion in low-voltage distribution grids, risking damage to network components, production losses and constraints on further PV deployment. Using a newly developed spatial expansion model for a Swiss subregion, this paper quantifies the timing and magnitude of grid-related losses in the near future. The results further illustrate how a small adjustment to existing subsidy schemes can delay binding grid constraints by one to two years and reduce congestion-related losses by up to 6.02 percent. While such a subsidy design does not represent a first-best economic solution, it offers a practical and readily implementable approach, addressing the coordination challenges associated with decentralized residential PV adoption.


## Main Execution Scripts
Both main execution scripts are located in [HPC_scenario_runs/](HPC_scenario_runs/) and are run from the repository root (e.g. `python HPC_scenario_runs/SCENARIOS.py`).
- The [SCENARIOS.py](HPC_scenario_runs/SCENARIOS.py) file serves as the primary entry point for running the PV allocation model. It defines a default scenario, derives all further scenarios from it (only the deviations from the default are written down, using `make_scenario`), and groups them into sub-scenario lists (e.g. `DEV`, `LRG`) that are selected with `get_subscen_list`. Each scenario is then executed by `PVAllocScenario` from [MAIN_pvallocation_Class.py](src/MAIN_pvallocation_Class.py) to simulate residential photovoltaic installation pathways across municipalities.
- The [VISUAL_ofSCENARIOS.py](HPC_scenario_runs/VISUAL_ofSCENARIOS.py) file handles post-processing and visualization tasks. It imports the same scenario lists from `SCENARIOS.py` and calls `Visualization` from [MAIN_visualization_Class.py](src/MAIN_visualization_Class.py) to generate plots, maps, and analytical outputs from the allocation results.

The remaining script files (`.py`, `.sh`, `.cmd`) in [HPC_scenario_runs/](HPC_scenario_runs/) are SLURM job scripts for model execution on the university's High-Performance Computing (HPC) cluster, enabling efficient processing of computationally intensive Monte Carlo iterations across multiple nodes. Older entry points (`main_INIT_array*.py`, `main_visualization.py`) are archived in `HPC_scenario_runs/before Oct26/`.

## Scenario Settings & Assumptions
[MAIN_pvallocation_Class.py](src/MAIN_pvallocation_Class.py) is structured using two classes, where the model execution is run in `PVAllocScenario`, serving all scenario settings from `PVAllocScenario_Settings`. This structure allows for flexible configuration of model parameters such as municipality selection, time horizons, technology specifications, and allocation algorithms while maintaining a clean separation between scenario configuration and execution logic.

Before running the PV allocation model, [MAIN_data_aggregation.py](src/MAIN_data_aggregation.py) must be executed first to transform all relevant data sources into a usable format. This preprocessing step integrates heterogeneous datasets including building registry data (GWR), grid topology information from the local distribution system operator, meteorological data, and PV installation records. The aggregation process standardizes coordinate systems, matches buildings to grid nodes, calculates solar radiation potentials, and prepares all necessary input files that the allocation model requires for its Monte Carlo simulations.

[calibration_class.py](src/calibration_randomforestreg/calibration_class.py) also needs to be executed first before any PV allocation models can be run. This file calibrates and stores a selection of prediction estimators used to determine the PV capacity that is installed on future houses in [MAIN_pvallocation_Class.py](src/MAIN_pvallocation_Class.py), determined by the settings `ALGOspec_pvinst_size_calculation` and `ALGOspec_calib_estim_mod_name_pkl`.

[MAIN_postanalysis.py](src/MAIN_postanalysis.py) contains additional analyses on finished PV allocation runs (e.g. worst grid nodes / worst weeks, battery buffers, and NPV-driven construction capacities).
<!-- 
## Code Structure and Process
For a better understanding, review the visualization of the model structure in the **[readme_OptimalPV_graph.png](docs/readme_OptimalPV_graph.png)**. The two files:
- [execution_scenarios.py](execution_scenarios.py) (defining all scenario settings (just deviations from default settings) for data aggregation and pv allocation model runs)
- [execution_RUNFILE.py](execution_RUNFILE.py) (the main file, running (selected parts of the model) with selected scenarios and visualization settings)

- **MASTER files:** The MASTER functions use docstrings to explain in a few steps what the called subfiles and functions are doing. Because they represent a larger process with many steps, MASTER functions do not return any objects but store various datafiles in an interm folder directory, which is then again accessed by the next MASTER function. This allows for development of only small specific parts in the entire model and later parallelization of the Monte Carlo iterations. 
- **Setting dictionaries:** Each MASTER file has a corresponding settings directory containing relevant information for the specific part of the model. These setting directories have default values, stored in the *default_settings.py* (e.g. [default_settings.py](pv_allocation/default_settings.py)) file of every code subfolder. For increased readabiligy, the run scenarios for the data aggregation and pv allocation step are defined in *execution_scenarios.py*. There only the changes to the default values are written down to increase visibility. The remaining default values for each scenario are later added with a function in [execution_RUNFILE.py](execution_RUNFILE.py).



For a detailed account of what the individual subfiles and functions are doing, read the [OptimalPV_RH_ModelDescription.pdf](docs/OptimalPV_RH_ModelDescription.pdf) which tries to stay up to date with all functions and their purpose. This model description will be transferred step by step to an individual README file in each subfolder. 
 -->
