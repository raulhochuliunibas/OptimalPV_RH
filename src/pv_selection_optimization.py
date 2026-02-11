"""
PV Selection Optimization Module

This module contains functions to optimize the selection of houses (EGIDs) for PV installation
such that the combined feed-in pattern does not exceed specified peak and cumulative constraints.

The optimization problem:
- Decision variables: Binary selection for each house (EGID)
- Objective: Maximize number of selected houses OR maximize total energy feed-in
- Constraints:
    1. Peak feed-in constraint: max(sum of hourly feed-in) <= peak_limit
    2. Cumulative feed-in constraint: sum(total feed-in) <= cumulative_limit
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional
import time


def prepare_feedin_matrix(node_subdf: pl.DataFrame) -> Tuple[Dict[str, int], np.ndarray, Dict]:
    """
    Prepare the feed-in data matrix from the polars dataframe.
    
    Parameters:
    -----------
    node_subdf : pl.DataFrame
        DataFrame with columns ['EGID', 'df_uid_winst', 'netfeedin_kW']
        Assumes data is sorted by EGID and time index
    
    Returns:
    --------
    egid_to_idx : Dict[str, int]
        Mapping from EGID to index in the matrix
    feedin_matrix : np.ndarray
        Matrix of shape (n_egids, n_hours) with feed-in values
    stats : Dict
        Statistics about the data preparation
    """
    print("Preparing feed-in matrix...")
    start_time = time.time()
    
    # Get unique EGIDs and create index mapping
    unique_egids = node_subdf['EGID'].unique().to_list()
    egid_to_idx = {egid: idx for idx, egid in enumerate(unique_egids)}
    n_egids = len(unique_egids)
    
    # Determine number of hours (assuming all EGIDs have same number of records)
    n_hours = node_subdf.filter(pl.col('EGID') == unique_egids[0]).height
    
    print(f"  Found {n_egids} unique EGIDs with {n_hours} time steps each")
    
    # Initialize matrix
    feedin_matrix = np.zeros((n_egids, n_hours))
    
    # Fill the matrix
    for egid in unique_egids:
        idx = egid_to_idx[egid]
        egid_data = node_subdf.filter(pl.col('EGID') == egid)['netfeedin_kW'].to_numpy()
        feedin_matrix[idx, :] = egid_data
        if egid == unique_egids[0]:
            pvprod_data = node_subdf.filter(pl.col('EGID') == egid)['pvprod_kW'].to_numpy()
            print(f' -- max pvprod_kW for     EGID {egid}: {np.max(pvprod_data):.2f} kW')
            print(f' -- sum pvprod_kWh for    EGID {egid}: {np.sum(pvprod_data):.2f} kWh')
            print(f' -- max netfeedin_kW for  EGID {egid}: {np.max(egid_data):.2f} kW')
            print(f' -- sum netfeedin_kWh for EGID {egid}: {np.sum(egid_data):.2f} kWh')

    stats = {
        'n_egids': n_egids,
        'n_hours': n_hours,
        'total_shape': feedin_matrix.shape,
        'preparation_time_sec': time.time() - start_time,
        'max_individual_peak_kW': np.max(feedin_matrix),
        'max_individual_annual_kWh': np.max(np.sum(feedin_matrix, axis=1)),
        'total_potential_peak_kW': np.sum(np.max(feedin_matrix, axis=1)),
        'total_potential_annual_kWh': np.sum(feedin_matrix),
    }
    
    print(f"  Matrix preparation completed in {stats['preparation_time_sec']:.2f} seconds")
    print(f"  Total potential peak: {stats['total_potential_peak_kW']:.2f} kW")
    print(f"  Total potential annual: {stats['total_potential_annual_kWh']:.2f} kWh")
    
    return egid_to_idx, feedin_matrix, stats


def optimize_pv_selection_greedy(
    feedin_matrix: np.ndarray,
    egid_to_idx: Dict[str, int],
    peak_limit_kW: float,
    cumulative_limit_kWh: Optional[float] = None,
    selection_strategy: str = 'peak_efficient',
    verbose: bool = True
) -> Tuple[List[str], Dict]:
    """
    Greedy optimization for PV selection.
    
    This is a heuristic approach that iteratively selects houses based on a selection strategy
    until constraints are violated. Much faster than exact optimization but not guaranteed optimal.
    
    Parameters:
    -----------
    feedin_matrix : np.ndarray
        Matrix of shape (n_egids, n_hours) with feed-in values
    egid_to_idx : Dict[str, int]
        Mapping from EGID to matrix index
    peak_limit_kW : float
        Maximum allowed peak feed-in in kW
    cumulative_limit_kWh : float, optional
        Maximum allowed cumulative feed-in in kWh
    selection_strategy : str
        Strategy for selecting houses:
        - 'peak_efficient': Select houses with lowest peak contribution
        - 'energy_max': Select houses with highest energy contribution
        - 'random': Random selection (for baseline comparison)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    selected_egids : List[str]
        List of selected EGIDs
    results : Dict
        Results and statistics
    """
    if verbose:
        print(f"\nStarting greedy optimization with strategy: {selection_strategy}")
        print(f"  Peak limit: {peak_limit_kW:.2f} kW")
        if cumulative_limit_kWh:
            print(f"  Cumulative limit: {cumulative_limit_kWh:.2f} kWh")
    
    start_time = time.time()
    
    # Create reverse mapping
    idx_to_egid = {idx: egid for egid, idx in egid_to_idx.items()}
    n_egids = feedin_matrix.shape[0]
    
    # Calculate metrics for each EGID
    peak_contribution = np.max(feedin_matrix, axis=1)
    annual_energy = np.sum(feedin_matrix, axis=1)
    
    # DYNAMIC GREEDY SELECTION (considers temporal complementarity)
    # Instead of pre-sorting, we evaluate each candidate against current selection
    selected_indices = []
    remaining_indices = set(range(n_egids))
    current_hourly_feedin = np.zeros(feedin_matrix.shape[1])
    current_peak = 0.0
    current_cumulative = 0.0
    
    if verbose:
        print(f"  Using dynamic selection to exploit temporal diversity...")
    
    while remaining_indices:
        best_idx = None
        best_score = -np.inf
        best_test_peak = None
        best_test_hourly = None
        
        # Evaluate all remaining houses
        for idx in remaining_indices:
            # Test adding this house
            test_hourly_feedin = current_hourly_feedin + feedin_matrix[idx, :]
            test_peak = np.max(test_hourly_feedin)
            test_cumulative = current_cumulative + annual_energy[idx]
            
            # Check constraints
            peak_ok = float(test_peak) <= peak_limit_kW
            cumulative_ok = (cumulative_limit_kWh is None) or (test_cumulative <= cumulative_limit_kWh)
            
            if peak_ok and cumulative_ok:
                # Score this candidate based on strategy
                if selection_strategy == 'peak_efficient':
                    # Maximize energy per peak capacity used
                    peak_margin_used = test_peak - current_peak
                    score = annual_energy[idx] / (peak_margin_used + 1e-6)
                elif selection_strategy == 'energy_max':
                    # Maximize energy added
                    score = annual_energy[idx]
                elif selection_strategy == 'random':
                    # Random selection among feasible
                    score = np.random.random()
                else:
                    score = 0
                
                # Track best candidate
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_test_peak = test_peak
                    best_test_hourly = test_hourly_feedin
        
        # If no feasible house found, stop
        if best_idx is None:
            break
        
        # Add best house to selection
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        current_hourly_feedin = best_test_hourly
        current_peak = best_test_peak
        current_cumulative += annual_energy[best_idx]
        
        if verbose and len(selected_indices) % max(1, n_egids // 20) == 0:
            print(f"    Selected {len(selected_indices)} houses, current peak: {current_peak:.2f} kW")
    
    # Convert indices back to EGIDs
    selected_egids = [idx_to_egid[idx] for idx in selected_indices]
    
    # Calculate final statistics
    final_hourly_feedin = np.sum(feedin_matrix[selected_indices, :], axis=0)
    
    results = {
        'selected_egids': selected_egids,
        'n_selected': len(selected_egids),
        'n_total': n_egids,
        'selection_rate': len(selected_egids) / n_egids,
        'final_peak_kW': np.max(final_hourly_feedin),
        'peak_limit_kW': peak_limit_kW,
        'peak_utilization': np.max(final_hourly_feedin) / peak_limit_kW,
        'final_cumulative_kWh': np.sum(final_hourly_feedin),
        'cumulative_limit_kWh': cumulative_limit_kWh,
        'cumulative_utilization': np.sum(final_hourly_feedin) / cumulative_limit_kWh if cumulative_limit_kWh else None,
        'optimization_time_sec': time.time() - start_time,
        'strategy': selection_strategy,
        'final_hourly_feedin': final_hourly_feedin,
    }
    
    if verbose:
        print(f"\nOptimization completed in {results['optimization_time_sec']:.2f} seconds")
        print(f"  Selected: {results['n_selected']} / {results['n_total']} houses ({results['selection_rate']*100:.1f}%)")
        print(f"  Final peak: {results['final_peak_kW']:.2f} kW (limit: {peak_limit_kW:.2f} kW, {results['peak_utilization']*100:.1f}% utilized)")
        print(f"  Final cumulative: {results['final_cumulative_kWh']:.2f} kWh", end='')
        if cumulative_limit_kWh:
            print(f" (limit: {cumulative_limit_kWh:.2f} kWh, {results['cumulative_utilization']*100:.1f}% utilized)")
        else:
            print(" (no limit)")
    
    return selected_egids, results


def optimize_pv_selection_pulp(
    feedin_matrix: np.ndarray,
    egid_to_idx: Dict[str, int],
    peak_limit_kW: float,
    cumulative_limit_kWh: Optional[float] = None,
    objective: str = 'maximize_count',
    time_limit_sec: Optional[int] = 300,
    verbose: bool = True,
    verbose_pulp_msg: bool = False,
    print_model_summary: bool = True,
) -> Tuple[List[str], Dict]:
    """
    Exact optimization using PuLP (Mixed Integer Linear Programming).
    
    This finds the optimal solution but may be slow for large problems.
    
    Parameters:
    -----------
    feedin_matrix : np.ndarray
        Matrix of shape (n_egids, n_hours) with feed-in values
    egid_to_idx : Dict[str, int]
        Mapping from EGID to matrix index
    peak_limit_kW : float
        Maximum allowed peak feed-in in kW
    cumulative_limit_kWh : float, optional
        Maximum allowed cumulative feed-in in kWh
    objective : str
        Optimization objective:
        - 'maximize_count': Maximize number of selected houses
        - 'maximize_energy': Maximize total energy feed-in
    time_limit_sec : int, optional
        Time limit for optimization in seconds (default: 300)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    selected_egids : List[str]
        List of selected EGIDs
    results : Dict
        Results and statistics
    """
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
    except ImportError:
        raise ImportError("PuLP is not installed. Install with: pip install pulp")
    
    if verbose:
        print(f"\nStarting exact optimization with PuLP")
        print(f"  Objective: {objective}")
        print(f"  Peak limit: {peak_limit_kW:.2f} kW")
        if cumulative_limit_kWh:
            print(f"  Cumulative limit: {cumulative_limit_kWh:.2f} kWh")
        print(f"  Time limit: {time_limit_sec} seconds")
    
    start_time = time.time()
    
    # Create reverse mapping
    idx_to_egid = {idx: egid for egid, idx in egid_to_idx.items()}
    n_egids, n_hours = feedin_matrix.shape
    
    # Calculate annual energy for each EGID
    annual_energy = np.sum(feedin_matrix, axis=1)
    
    # Create the optimization problem
    prob = LpProblem("PV_Selection", LpMaximize)
    
    # Decision variables: binary selection for each EGID
    select = [LpVariable(f"select_{idx}", cat='Binary') for idx in range(n_egids)]
    
    # Objective function
    if objective == 'maximize_count':
        # Maximize number of selected houses
        prob += lpSum(select)
    elif objective == 'maximize_energy':
        # Maximize total energy
        prob += lpSum([select[idx] * annual_energy[idx] for idx in range(n_egids)])
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Constraint 1: Peak feed-in limit (for each hour)
    if verbose:
        print(f"  Adding {n_hours} peak constraints...")
    
    for hour in range(n_hours):
        hour_feedin = lpSum([select[idx] * feedin_matrix[idx, hour] for idx in range(n_egids)])
        prob += hour_feedin <= peak_limit_kW, f"Peak_constraint_hour_{hour}"
    
    # Constraint 2: Cumulative feed-in limit (optional)
    if cumulative_limit_kWh is not None:
        if verbose:
            print(f"  Adding cumulative constraint...")
        total_energy = lpSum([select[idx] * annual_energy[idx] for idx in range(n_egids)])
        prob += total_energy <= cumulative_limit_kWh, "Cumulative_constraint"
    
    # Solve the problem
    if verbose:
        print(f"  Solving optimization problem...")

        if print_model_summary:
            try:
                def _iter_affine_terms(expr):
                    try:
                        items = list(expr.items())
                    except Exception:
                        items = []
                    terms = []
                    for var, coeff in items:
                        name = getattr(var, 'name', str(var))
                        try:
                            coeff_f = float(coeff)
                        except Exception:
                            coeff_f = coeff
                        terms.append((name, coeff_f))
                    return terms

                def _summarize_affine(expr, max_terms=6):
                    terms = _iter_affine_terms(expr)
                    # sort by absolute coefficient descending for better signal
                    try:
                        terms_sorted = sorted(terms, key=lambda t: abs(t[1]) if isinstance(t[1], (int, float)) else 0, reverse=True)
                    except Exception:
                        terms_sorted = terms
                    shown = terms_sorted[:max_terms]
                    parts = [f"{coeff:g}*{name}" for name, coeff in [(n, c) for (n, c) in shown]]
                    more = len(terms_sorted) - len(shown)
                    tail = f" (+{more} more)" if more > 0 else ""
                    # constant term if present
                    const = getattr(expr, 'constant', 0)
                    if const:
                        parts.append(f"{const:g}")
                    return " + ".join(parts) + tail

                def _sense_symbol(sense_val):
                    # PuLP uses: -1 (<=), 0 (==), 1 (>=)
                    try:
                        return { -1: "<=", 0: "==", 1: ">=" }.get(int(getattr(sense_val, 'value', sense_val)), "?")
                    except Exception:
                        return "?"

                # Collect counts
                n_vars = len(prob.variables())
                n_cons = len(getattr(prob, 'constraints', {}))
                obj_str = _summarize_affine(prob.objective, max_terms=8)
                print("\n  Model summary (truncated):")
                print(f"    Variables: {n_vars}")
                print(f"    Constraints: {n_cons}")
                print(f"    Objective: {obj_str}")

                # Show a few constraints
                cons_items = list(getattr(prob, 'constraints', {}).items())
                max_show = min(4    , len(cons_items))
                if max_show:
                    print(f"    Sample constraints (first {max_show}):")
                    for i in range(max_show):
                        cname, c = cons_items[i]
                        expr = getattr(c, 'e', c)
                        lhs = _summarize_affine(expr, max_terms=6)
                        rhs = getattr(c, 'constant', None)
                        sense = _sense_symbol(getattr(c, 'sense', None))
                        rhs_part = f" {sense} {rhs:g}" if isinstance(rhs, (int, float, np.floating)) else f" {sense} ?"
                        print(f"      {cname}: {lhs}{rhs_part}")
            except Exception as _print_err:
                # Avoid failing the solve due to summary printing
                print(f"    (Model summary unavailable: {type(_print_err).__name__})")
    
    solver = PULP_CBC_CMD(msg=verbose_pulp_msg, timeLimit=time_limit_sec)
    prob.solve(solver)
    
    # Extract results
    selected_indices = [idx for idx in range(n_egids) if select[idx].varValue > 0.5]
    selected_egids = [idx_to_egid[idx] for idx in selected_indices]
    
    # Calculate final statistics
    final_hourly_feedin = np.sum(feedin_matrix[selected_indices, :], axis=0)
    
    results = {
        'selected_egids': selected_egids,
        'n_selected': len(selected_egids),
        'n_total': n_egids,
        'selection_rate': len(selected_egids) / n_egids,
        'final_peak_kW': np.max(final_hourly_feedin) if len(selected_indices) > 0 else 0,
        'peak_limit_kW': peak_limit_kW,
        'peak_utilization': np.max(final_hourly_feedin) / peak_limit_kW if len(selected_indices) > 0 else 0,
        'final_cumulative_kWh': np.sum(final_hourly_feedin),
        'cumulative_limit_kWh': cumulative_limit_kWh,
        'cumulative_utilization': np.sum(final_hourly_feedin) / cumulative_limit_kWh if cumulative_limit_kWh and len(selected_indices) > 0 else None,
        'optimization_time_sec': time.time() - start_time,
        'objective': objective,
        'solver_status': LpStatus[prob.status],
        'final_hourly_feedin': final_hourly_feedin,
    }
    
    if verbose:
        print(f"\nOptimization completed in {results['optimization_time_sec']:.2f} seconds")
        print(f"  Solver status: {results['solver_status']}")
        print(f"  Selected: {results['n_selected']} / {results['n_total']} houses ({results['selection_rate']*100:.1f}%)")
        print(f"  Final peak: {results['final_peak_kW']:.2f} kW (limit: {peak_limit_kW:.2f} kW, {results['peak_utilization']*100:.1f}% utilized)")
        print(f"  Final cumulative: {results['final_cumulative_kWh']:.2f} kWh", end='')
        if cumulative_limit_kWh:
            print(f" (limit: {cumulative_limit_kWh:.2f} kWh, {results['cumulative_utilization']*100:.1f}% utilized)")
        else:
            print(" (no limit)")
    
    return selected_egids, results


def run_pv_selection_optimization(
    node_subdf: pl.DataFrame,
    peak_limit_kW: float,
    cumulative_limit_kWh: Optional[float] = None,
    method: str = 'greedy',
    **kwargs
) -> Tuple[List[str], pl.DataFrame, Dict]:
    """
    Main function to run PV selection optimization.
    
    Parameters:
    -----------
    node_subdf : pl.DataFrame
        DataFrame with columns ['EGID', 'df_uid_winst', 'netfeedin_kW']
    peak_limit_kW : float
        Maximum allowed peak feed-in in kW
    cumulative_limit_kWh : float, optional
        Maximum allowed cumulative feed-in in kWh
    method : str
        Optimization method: 'greedy' or 'pulp'
    **kwargs : additional arguments passed to the optimization function
    
    Returns:
    --------
    selected_egids : List[str]
        List of selected EGIDs
    selected_df : pl.DataFrame
        Filtered dataframe with only selected EGIDs
    results : Dict
        Optimization results and statistics
    """
    # Prepare data
    egid_to_idx, feedin_matrix, prep_stats = prepare_feedin_matrix(node_subdf)
    
    # Run optimization
    if method == 'greedy':
        selected_egids, results = optimize_pv_selection_greedy(
            feedin_matrix, egid_to_idx, peak_limit_kW, cumulative_limit_kWh, **kwargs
        )
    elif method == 'pulp':
        selected_egids, results = optimize_pv_selection_pulp(
            feedin_matrix, egid_to_idx, peak_limit_kW, cumulative_limit_kWh, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'pulp'")
    
    # Add preparation stats to results
    results['preparation_stats'] = prep_stats
    
    # Filter original dataframe to selected EGIDs
    selected_df = node_subdf.filter(pl.col('EGID').is_in(selected_egids))
    
    return selected_egids, selected_df, results


# Example usage
if __name__ == "__main__":
    # Example: Create dummy data
    print("Creating example data...")
    
    # Simulate data for 100 houses over 8760 hours (one year)
    n_houses = 100
    n_hours = 8760
    
    # Create synthetic feed-in patterns
    np.random.seed(42)
    egids = [f"EGID_{i:05d}" for i in range(n_houses)]
    
    # Create hourly data
    data_list = []
    for egid in egids:
        # Random capacity between 5-15 kW
        capacity = np.random.uniform(5, 15)
        
        # Simplified daily pattern (solar generation)
        hourly_pattern = []
        for hour in range(n_hours):
            hour_of_day = hour % 24
            day_of_year = hour // 24
            
            # Sinusoidal daily pattern (peak at noon)
            if 6 <= hour_of_day <= 18:
                time_factor = np.sin((hour_of_day - 6) * np.pi / 12)
            else:
                time_factor = 0
            
            # Seasonal variation
            season_factor = 0.5 + 0.5 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            
            # Random weather variation
            weather_factor = np.random.uniform(0.7, 1.0)
            
            feedin = capacity * time_factor * season_factor * weather_factor
            hourly_pattern.append(feedin)
            
        for hour, feedin_val in enumerate(hourly_pattern):
            data_list.append({
                'EGID': egid,
                'df_uid_winst': f"{egid}_roof1",
                'netfeedin_kW': feedin_val
            })
    
    # Create polars dataframe
    node_subdf = pl.DataFrame(data_list)
    
    print(f"\nCreated synthetic data:")
    print(f"  {len(egids)} houses")
    print(f"  {n_hours} hours")
    print(f"  Total rows: {len(data_list)}")
    
    # Run optimization with different methods
    print("\n" + "="*80)
    print("GREEDY OPTIMIZATION (peak-efficient strategy)")
    print("="*80)
    
    selected_egids_greedy, selected_df_greedy, results_greedy = run_pv_selection_optimization(
        node_subdf=node_subdf,
        peak_limit_kW=300,  # 300 kW peak limit
        cumulative_limit_kWh=None,
        method='greedy',
        selection_strategy='peak_efficient',
        verbose=True
    )
    
    print("\n" + "="*80)
    print("GREEDY OPTIMIZATION (energy-max strategy)")
    print("="*80)
    
    selected_egids_greedy2, selected_df_greedy2, results_greedy2 = run_pv_selection_optimization(
        node_subdf=node_subdf,
        peak_limit_kW=300,
        cumulative_limit_kWh=None,
        method='greedy',
        selection_strategy='energy_max',
        verbose=True
    )
    
    print("\n" + "="*80)
    print("EXACT OPTIMIZATION (requires PuLP)")
    print("="*80)
    
    try:
        selected_egids_pulp, selected_df_pulp, results_pulp = run_pv_selection_optimization(
            node_subdf=node_subdf,
            peak_limit_kW=300,
            cumulative_limit_kWh=None,
            method='pulp',
            objective='maximize_count',
            time_limit_sec=60,
            verbose=True
        )
    except ImportError as e:
        print(f"PuLP optimization skipped: {e}")
