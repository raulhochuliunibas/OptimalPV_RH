# python integrated packages
import sys
import os as os
import glob
import json
import shutil

# external packages
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.visualization.visual_core import VisualizationCore


class VisualizationExportUtils(VisualizationCore):
    """Non-plot export helpers: copy plot-data files for the static plots, write the HPC cmd file."""

    def export_cmd_GridOptim_parallel_HPC(self, name_dir_export): 

        topo = json.load(open(f'{self.visual_sett.data_path}/pvalloc/{name_dir_export}/topo_egid.json', 'r'))
        topo_df_list = []
        for k,v in topo.items():
            rows = {
                'EGID': k,
                'grid_node': v.get('grid_node'),
            }
            topo_df_list.append(rows)
        topo_df = pl.DataFrame(topo_df_list)

        topo_df = topo_df.group_by('grid_node').agg([
            pl.count('EGID').alias('n_egids_in_node'),
        ])
        topo_df = topo_df.sort('n_egids_in_node', descending=True)
        topo_df = topo_df.with_row_index(
            name="grid_node_idx",
            offset=1
        )

        # create and export txt lines for cmd file (parallelization HPC)
        lines = (
            topo_df
            .select(
                pl.format(
                    "python main_OPTIM_array_LRG.py {}",
                    pl.col("grid_node")
                ).alias("cmd")
            )
            .to_series()
            .to_list()
        )

        with open(f'{self.visual_sett.wd_path}/main_OPTIM_array_LRG_XYh_XcXG.txt', 'w') as f:
            for line in lines:
                f.write(f"{line}\n")


    def export_csvs_for_static_plots(self, file_path_tuple_list=[
        ('zMC1', 'pred_npv_inst_by_M', 'npv_df_1.parquet'),
    ], export_dir_path=None):
        # Export scenario-specific files to the static visualization folder.
        # Each tuple is interpreted as a path relative to data/pvalloc/<scen>/.
        export_dir_path = self.visual_sett.visual_static_export_path if export_dir_path is None else export_dir_path
        os.makedirs(export_dir_path, exist_ok=True)

        for scen in self.visual_sett.pvalloc_scen_list:
            self.get_pvalloc_sett_output(pvalloc_scen_name=scen)
            scen_root_path = os.path.join(self.visual_sett.data_path, 'pvalloc', scen)

            for path_tuple in file_path_tuple_list:
                if isinstance(path_tuple, (str, bytes)):
                    path_tuple = (path_tuple,)
                if len(path_tuple) == 0:
                    continue

                path_tuple = tuple(str(part) for part in path_tuple if part not in (None, ''))
                if len(path_tuple) == 0:
                    continue

                source_name = path_tuple[-1]
                path_origin = os.path.join(scen_root_path, *path_tuple)

                candidate_paths = []
                if os.path.isfile(path_origin):
                    candidate_paths.append(path_origin)
                else:
                    fallback_matches = glob.glob(
                        os.path.join(scen_root_path, '**', source_name),
                        recursive=True,
                    )
                    candidate_paths.extend(
                        match for match in sorted(fallback_matches)
                        if os.path.isfile(match)
                    )

                if not candidate_paths:
                    print(f'Skipping missing export source for scen={scen}: {"/".join(path_tuple)}')
                    continue

                path_origin = candidate_paths[0]
                file_stem, file_ext = os.path.splitext(os.path.basename(path_origin))
                path_dest = os.path.join(export_dir_path, f'{file_stem}_{scen}{file_ext}')

                if os.path.abspath(path_origin) == os.path.abspath(path_dest):
                    continue

                try:
                    shutil.copy2(path_origin, path_dest)
                    print(f'Copied CSV for static plots: {path_origin} -> {path_dest}')
                except FileNotFoundError:
                    print(f'Skipping missing export source for scen={scen}: {path_origin}')
                except OSError as exc:
                    print(f'Could not copy export source for scen={scen}: {path_origin} -> {path_dest} ({exc})')
