import os
import sys
from dataclasses import replace
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization
from main_INIT_array import get_subscen_list

if __name__ == "__main__":

    pvalloc_scen_list = get_subscen_list('LRG')

    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID_ENV', 'unknown')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID_ENV', 'unknown')
    slurm_full_id = f"{slurm_job_id}_{slurm_array_id}"

    if len(sys.argv) > 1:
        pvalloc_scen_index = int(sys.argv[1])
        if pvalloc_scen_index < len(pvalloc_scen_list):
            pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
            scen_class = PVAllocScenario(pvalloc_scen)

            scen_class.sett.slurm_full_id        = slurm_full_id
            scen_class.sett.pvalloc_scen_index   = pvalloc_scen_index
    
            scen_class.run_pvalloc_initalization()
            scen_class.run_pvalloc_mcalgorithm()
    
        print('done')
        