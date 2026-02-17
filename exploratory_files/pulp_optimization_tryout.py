import pulp
import numpy as np
import pandas as pd
import plotly.graph_objects as go



# -------------------------
# Data
# -------------------------

np.random.seed(0)

hours = range(24)
pv_profiles = {
    "PV1": np.random.uniform(0, 5, 24),
    "PV2": np.random.uniform(0, 6, 24),
    "PV3": np.random.uniform(0, 4, 24),
    "PV4": np.random.uniform(0, 9, 24),
    "PV5": np.random.uniform(0, 3, 24),
}

# Grid node capacity (MW) for each hour
grid_capacity = np.full(24, 13)

# -------------------------
# Model
# -------------------------

model = pulp.LpProblem("PV_Selection", pulp.LpMaximize)

# Binary decision variables
x = {
    pv: pulp.LpVariable(f"select_{pv}", cat="Binary")
    for pv in pv_profiles
}

# -------------------------
# Objective: maximize total energy
# -------------------------

model += pulp.lpSum(
    pv_profiles[pv][h] * x[pv]
    for pv in pv_profiles
    for h in hours
)

# -------------------------
# Constraints
# -------------------------

# Grid capacity per hour
for h in hours:
    model += (
        pulp.lpSum(pv_profiles[pv][h] * x[pv] for pv in pv_profiles)
        <= grid_capacity[h],
        f"GridCapacity_hour_{h}"
    )

# Only 2 PV systems allowed
model += pulp.lpSum(x[pv] for pv in pv_profiles) <=5

# -------------------------
# Solve
# -------------------------

model.solve(pulp.PULP_CBC_CMD(msg=False))

# -------------------------
# Results
# -------------------------

print("Selected PV systems:")
for pv in pv_profiles:
    print(f"  {pv}: {int(x[pv].value())}")

print("\nTotal energy injected:",
      pulp.value(model.objective))

print('end')

print('\nPlotting results...')
print(f'pv1-3: {sum(pv_profiles["PV1"]) + sum(pv_profiles["PV2"]) + sum(pv_profiles["PV3"])} MW')

print(f'pv1,4: {sum(pv_profiles["PV1"]) + sum(pv_profiles["PV4"])} MW')
print(f'pv2,4: {sum(pv_profiles["PV2"]) + sum(pv_profiles["PV4"])} MW')

# Determine selection order based on total energy among selected PVs
selected_pvs = [pv for pv in pv_profiles ]
pv_sums = {pv: float(np.sum(pv_profiles[pv])) for pv in selected_pvs}
ordered = sorted(selected_pvs, key=lambda pv: pv_sums[pv], reverse=True)
pv_order = {pv: i + 1 for i, pv in enumerate(ordered)}

if ordered:
    print("\nSelection order (1 = highest total energy among selected):")
    for i, pv in enumerate(ordered, start=1):
        print(f"  {i}. {pv} ({pv_sums[pv]:.1f} MW)")
else:
    print("\nNo PV systems selected.")

fig = go.Figure()
for pv in pv_profiles:
    fig.add_trace(go.Scatter(x=list(hours), 
                                y=pv_profiles[pv],
                                mode='lines+markers',
                                stackgroup="one",     # enables stacking
                                line=dict(width=2), 
                                name=f'pv: {pv}, x: {int(x[pv].value())}, order: {pv_order.get(pv, "-")}, sum: {sum(pv_profiles[pv]):.1f} MW'))
fig.add_trace(go.Scatter(x=list(hours), 
                         y=grid_capacity,
                            mode='lines',
                            line=dict(width=4, color='black', dash='dash'),
                            name=f'Grid Capacity ({max(grid_capacity)} MW)'))
fig.update_layout(title='Selected PV Profiles vs Grid Capacity',
                    xaxis_title='Hour of Day',
                    yaxis_title='Power (MW)', 
                    template='plotly_white')
fig.show()