# HPC Scenario Run Scripts

This directory contains all High-Performance Computing (HPC) job submission scripts and scenario initiation files for the OptimalPV_RH project.

## File Organization

- **`.sh` files**: Shell scripts for Linux/Unix HPC cluster submission (e.g., SLURM)
- **`.cmd` files**: Windows batch scripts for job submission
- **`.py` files**: Python scripts that initiate specific scenario runs

## Use Cases

These scripts are typically used to:
- Initialize calibration runs (e.g., `main_INIT_array_*.py`)
- Run optimization scenarios (e.g., `main_OPTIM_*.py`)
- Execute data aggregation tasks (e.g., `data_agg_*.sh`)
- Generate visualizations (e.g., `visual_*.sh`)
- Allocate PV resources (e.g., `pvalloc_*.sh`)

## Running Scenarios

For details on how to execute these scripts, refer to the main project README.
