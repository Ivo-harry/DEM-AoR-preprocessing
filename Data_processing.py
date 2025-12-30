# -----------------------------------------------------------
# LIBRARIES IMPORTS
# -----------------------------------------------------------

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------------------------------------
# FILES, MATRIXES, PARAMETERS SET-UP
# -----------------------------------------------------------

files = {
    "0.05 friction": "X_Z_particle_pos_005friction.csv",
    "0.08 friction": "X_Z_particle_pos_008friction.csv",
    "0.10 friction": "X_Z_particle_pos_010friction.csv",
    "0.15 friction": "X_Z_particle_pos_015friction.csv",
    "0.25 friction": "X_Z_particle_pos_025friction.csv"
}
base_dir_reproduceability = "Initial Study reproduceability check"


files_velocity = {
    "0.05 friction": "Velocity_fric_005",
    "0.08 friction": "Velocity_fric_008",
    "0.10 friction": "Velocity_fric_010",
    "0.15 friction": "Velocity_fric_015",
    "0.25 friction": "Velocity_fric_025"
}
base_dir_velocty = "Velocity Data"

files_timestep = {
    "0.20 timestep": "Kin_enr_Vel_20_perc",
    "0.40 timestep": "Kin_enr_Vel_40_perc",
    "0.60 timestep": "Kin_enr_Vel_60_perc",
    "0.80 timestep": "Kin_enr_Vel_80_perc",
    "1.00 timestep": "Kin_enr_Vel_100_perc"
}
DATA_DIR = "Timestep Data"

files_Significance = {
    "1": "Run_1",
    "2": "Run_2",
    "3": "Run_3",
    "4": "Run_4",
    "5": "Run_5",
    "6": "Run_6",
    "7": "Run_7",
    "8": "Run_8",
    "9": "Run_9",
    "10": "Run_10",
    "11": "Run_11",
    "12": "Run_12",
}
dir_doe_significance = "Parameters Final"

files_Statistics = {
    "1": "Run_1",
    "2": "Run_2",
    "3": "Run_3",
    "4": "Run_4",
    "5": "Run_5",
    "6": "Run_6",
    "7": "Run_7",
    "8": "Run_8",
    "9": "Run_9",
    "10": "Run_10",
    "11": "Run_11",
    "12": "Run_12",
    "13": "Run_13",
    "14": "Run_14",
    "15": "Run_15",
    "16": "Run_16",
    "17": "Run_17",
    "18": "Run_18",
    "19": "Run_19",
    "20": "Run_20",
    "21": "Run_21",
    "22": "Run_22",
    "23": "Run_23",
    "24": "Run_24",
    "25": "Run_25"
}
dir_Final_Doe = "Final DoE results"

# Parameter values for each level (0–4)
param_values = {
    "A_restitution_pp":         [0.15, 0.275, 0.4, 0.525, 0.65],
    "B_static_friction_pp":     [0.3, 0.425, 0.55, 0.675, 0.8],
    "C_rolling_friction_pp":    [0.05, 0.1375, 0.225, 0.3125, 0.4],
}

runs = [
    {"run": 1,  "A": 0, "B": 0, "C": 0},
    {"run": 2,  "A": 0, "B": 1, "C": 2},
    {"run": 3,  "A": 0, "B": 2, "C": 4},
    {"run": 4,  "A": 0, "B": 3, "C": 1},
    {"run": 5,  "A": 0, "B": 4, "C": 3},
    {"run": 6,  "A": 1, "B": 0, "C": 4},
    {"run": 7,  "A": 1, "B": 1, "C": 1},
    {"run": 8,  "A": 1, "B": 2, "C": 3},
    {"run": 9,  "A": 1, "B": 3, "C": 0},
    {"run": 10, "A": 1, "B": 4, "C": 2},
    {"run": 11, "A": 2, "B": 0, "C": 3},
    {"run": 12, "A": 2, "B": 1, "C": 0},
    {"run": 13, "A": 2, "B": 2, "C": 2},
    {"run": 14, "A": 2, "B": 3, "C": 4},
    {"run": 15, "A": 2, "B": 4, "C": 1},
    {"run": 16, "A": 3, "B": 0, "C": 2},
    {"run": 17, "A": 3, "B": 1, "C": 4},
    {"run": 18, "A": 3, "B": 2, "C": 1},
    {"run": 19, "A": 3, "B": 3, "C": 3},
    {"run": 20, "A": 3, "B": 4, "C": 0},
    {"run": 21, "A": 4, "B": 0, "C": 1},
    {"run": 22, "A": 4, "B": 1, "C": 3},
    {"run": 23, "A": 4, "B": 2, "C": 0},
    {"run": 24, "A": 4, "B": 3, "C": 2},
    {"run": 25, "A": 4, "B": 4, "C": 4},
]


signs = pd.DataFrame({
    "A":  [-1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1],
    "B":  [-1, -1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1],
    "C":  [-1, -1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1],
    "D":  [ 1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1], 
    "E":  [-1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1],
    "F":  [ 1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1,  1],
    "G":  [ -1, 1, -1,  1, -1,  1, -1,  1, -1,  1,  1,  1],
    "H":  [ 1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1],
    "I":  [ 1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1],
    "J":  [ 1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1,  1],
    "K":  [ 1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1,  1]
})


# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------

def compute_angle_of_repose_part_significance(csv_file, visualize=True, VISUALIZE=False):
    # --- Load the CSV ---
    with open(csv_file, "r") as f:
        lines = f.readlines()

    last_x_line = None
    last_z_line = None

    # Search from bottom to top for the last X and Z entries
    for line in reversed(lines):
        if last_z_line is None and line.startswith("Q02 : Particle Position Z"):
            last_z_line = line
        elif last_x_line is None and line.startswith("Q01 : Particle Position X"):
            last_x_line = line
        
        if last_x_line and last_z_line:
            break

    if last_x_line is None or last_z_line is None:
        raise ValueError("Could not find X/Z particle data in the file!")

    # Parse X and Z arrays
    x = np.array([float(v) for v in last_x_line.split(",")[1:]])
    y = np.array([float(v) for v in last_z_line.split(",")[1:]])

    # -----------------------------------------------------------
    # PREPROCESSING THE INCLUDED ORIGINAL POINTS DATASEET FOR THE VISUALIZATION
    # -----------------------------------------------------------

    # Keep only particles with non-negative Z

    mask = (x >= 458) & (y >= 0)
    x = x[mask]
    y = y[mask]
    

    # -----------------------------------------------------------
    # RAW UPPER ENVELOPE
    # -----------------------------------------------------------
    num_bins = 45
    bins = np.linspace(x.min(), x.max(), num_bins + 1)

    bin_centers = []
    bin_max_y = []

    for i in range(num_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_max_y.append(y[mask].max())

    bin_centers = np.array(bin_centers).reshape(-1, 1)
    bin_max_y = np.array(bin_max_y)


    # -----------------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------------
    threshold_flat = 0.05   # absolute difference threshold for flat tail
    spike_factor = 1.5      # factor above moving average to consider as spike
    window_size = 5         # moving average window (odd number)
    half_window = window_size // 2

    # -------------------------------
    # INPUT DATA (example)
    # -------------------------------
    # bin_centers = np.array([...])
    # bin_max_y = np.array([...])

    bin_centers = np.array(bin_centers)
    bin_max_y = np.array(bin_max_y)

    # -------------------------------
    # STEP 1: Remove first bin
    # -------------------------------
    bin_centers = bin_centers[1:]
    bin_max_y = bin_max_y[1:]

    # -------------------------------
    # STEP 2: Remove flat tails robustly
    # -------------------------------
    flat_centers = []
    flat_max_y = []

    slope_threshold = 0.02      # tune this (y-units per x-unit)
    tail_y_fraction = 0.15      # tail only allowed below 15% of max
    window = 3

    max_y_global = np.max(bin_max_y)

    i = 0
    tail_started = False

    while i < len(bin_max_y):

        if i + window >= len(bin_max_y):
            break

        # ---- Flat-value check (your existing logic) ----
        start_val = bin_max_y[i]
        j = i + 1
        while j < len(bin_max_y) and abs(bin_max_y[j] - start_val) < threshold_flat:
            j += 1

        if j - i > 1:
            i = j
            continue

        # ---- Slope check ----
        dx = bin_centers[i + window] - bin_centers[i]
        dy = bin_max_y[i + window] - bin_max_y[i]
        slope = dy / dx

        # ---- Tail detection condition ----
        if (
            abs(slope) < slope_threshold and
            bin_max_y[i] < tail_y_fraction * max_y_global
        ):
            # Tail reached → stop keeping points
            break

        # ---- Keep point ----
        flat_centers.append(bin_centers[i])
        flat_max_y.append(bin_max_y[i])
        i += 1

    flat_centers = np.array(flat_centers)
    flat_max_y = np.array(flat_max_y)

    # -------------------------------
    # STEP 3: Remove spikes using moving average
    # -------------------------------
    clean_centers = []
    clean_max_y = []

    for i in range(len(flat_max_y)):
        # moving average window
        start = max(0, i - half_window)
        end = min(len(flat_max_y), i + half_window + 1)
        window_avg = np.mean(flat_max_y[start:end])
        
        # remove spikes
        if flat_max_y[i] > spike_factor * window_avg:
            continue
        
        clean_centers.append(flat_centers[i])
        clean_max_y.append(flat_max_y[i])

    clean_centers = np.array(clean_centers).reshape(-1, 1)
    clean_max_y = np.array(clean_max_y)

    # Maximum Z from cleaned bins
    max_z_bin = clean_max_y.max() if len(clean_max_y) > 0 else None



    # -----------------------------------------------------------
    # LINEAR REGRESSION
    # -----------------------------------------------------------
    if len(clean_centers) > 0:
        model = LinearRegression().fit(clean_centers, clean_max_y)
        slope = model.coef_[0]
        angle_deg = abs(np.degrees(np.arctan(slope)))
    else:
        angle_deg = None

    # -----------------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------------

    if VISUALIZE:
        plt.figure(figsize=(10, 6))
        # Particles
        plt.scatter(x, y, color='blue', alpha=0.3, s=10, label='Particles')
        # Original envelope
        plt.scatter(bin_centers, bin_max_y, color='gray', s=50, alpha=0.5, label='Envelope bins')
        # Cleaned envelope
        plt.scatter(clean_centers, clean_max_y, color='green', s=60, label='Cleaned envelope')
        # Regression line
        x_fit = np.linspace(clean_centers.min(), clean_centers.max(), 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)
        plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Linear fit')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Angle of repose: {angle_deg:.2f}°" if angle_deg else "No valid slope")
        plt.legend()
        plt.grid(True)
        plt.show()

    return angle_deg, max_z_bin

def load_velocity_files(file_dict, start_time=1.25, base_dir="Velocity Data"):
    """
    Reads multiple velocity text files, combines them into a single DataFrame,
    filters starting from start_time, and interpolates missing values for smooth lines.
    """
    df_combined = pd.DataFrame()
    
    for label, filename in file_dict.items():
        times = []
        velocities = []

        full_path = os.path.join(base_dir, filename)
        assert os.path.exists(full_path), f"Missing file: {full_path}"

        with open(full_path, 'r') as f:
            lines = f.readlines()
            data_start = False

            for line in lines:
                line = line.strip()

                if line.startswith("EXTRACTED DATA"):
                    data_start = True
                    continue

                if data_start:
                    if line.startswith("TIME:"):
                        times.append(float(line.split(",")[1]))
                    elif line.startswith("Q01"):
                        velocities.append(float(line.split(",")[1]))

        temp_df = pd.DataFrame({
            "Time": times,
            label: velocities
        })

        if df_combined.empty:
            df_combined = temp_df
        else:
            df_combined = pd.merge(df_combined, temp_df, on="Time", how="outer")
    
    # Sort by time
    df_combined = df_combined.sort_values("Time").reset_index(drop=True)
    
    # Filter to start from start_time
    df_combined = df_combined[df_combined["Time"] >= start_time].reset_index(drop=True)
    
    # Interpolate missing values for smooth lines
    df_combined = df_combined.interpolate(method="linear")
    
    return df_combined

def parse_file(path):
    times = []
    kinetic_energy = []

    with open(path, "r") as f:
        lines = f.readlines()

    current_time = None

    for line in lines:
        line = line.strip()

        # TIME entry
        if line.startswith("TIME:,"):
            try:
                current_time = float(line.split(",")[1])
            except:
                current_time = None

        # Q01 entry (kinetic energy)
        elif line.startswith("Q01 : Average Particle Kinetic Energy:,"):
            if current_time is None:
                continue
            try:
                ke = float(line.split(",")[1])
                times.append(current_time)
                kinetic_energy.append(ke)
            except:
                pass

    return times, kinetic_energy

# --------------------------------------------------------------------
# PLOTTING FUNCTIONS FOR DIFFERENT PARTS
# --------------------------------------------------------------------

# Function to plot
def plot_velocity(df, title="Average Particle Velocity vs Time"):
    plt.figure(figsize=(10,6))
    for col in df.columns:
        if col != "Time":
            plt.plot(df["Time"], df[col], label=col)  # smooth connected lines
    plt.xlabel("Time [s]")
    plt.ylabel("Average Particle Velocity [m/s]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def make_plot(times, ke, title, output_name):
    plt.figure(figsize=(8, 5))
    plt.plot(times, ke)
    plt.xlabel("Time [s]")
    plt.ylabel("Average Particle Kinetic Energy [J]")
    plt.title(f"Kinetic Energy vs Time ({title})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

def make_combined_plot(data_dict, output_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (label, (times, ke)) in enumerate(data_dict.items()):
        ax = axes[idx]
        ax.plot(times, ke)
        ax.set_title(f"{label}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Kinetic Energy [J]")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

def plot_expected_vs_actual(
    df, 
    actual_col, 
    expected_col, 
    ylabel="Angle of Repose (deg)", 
    test_idx=None,
    test_color="red",
    train_color="blue",
    actual_color="yellow"
):
    """
    Plots Expected vs Actual values as points, sorted by the actual values (ascending).
    Actual values are all same color.
    Optionally, training vs test predicted points are colored differently.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    actual_col : str
        Column name of actual values
    expected_col : str
        Column name of expected/predicted values
    ylabel : str, optional
        Y-axis label
    test_idx : list-like or None, optional
        Indices of test points to highlight. Default is None (all predicted points same color)
    test_color : str, optional
        Color for predicted test points
    train_color : str, optional
        Color for predicted training points
    actual_color : str, optional
        Color for actual values
    """
    df_sorted = df.sort_values(actual_col).reset_index(drop=True)
    #df_sorted = df
    x = range(len(df_sorted))
    
    plt.figure()
    
    # Plot all actual points in one color
    plt.scatter(x, df_sorted[actual_col], color=actual_color, label=f"Actual ({actual_col})")
    
    if test_idx is None:
        # All predicted points same color
        plt.scatter(x, df_sorted[expected_col], color=train_color, label=f"Predicted ({expected_col})")
    else:
        # Split predicted points by train/test
        test_mask = df_sorted.index.isin(test_idx)
        train_mask = ~test_mask
        
        plt.scatter(
            [i for i, m in zip(x, train_mask) if m],
            df_sorted.loc[train_mask, expected_col],
            color=train_color,
            label=f"Predicted ({expected_col}) - train"
        )
        plt.scatter(
            [i for i, m in zip(x, test_mask) if m],
            df_sorted.loc[test_mask, expected_col],
            color=test_color,
            label=f"Predicted ({expected_col}) - test"
        )
    
    plt.xlabel(f"Run (sorted by {actual_col})")
    plt.ylabel(ylabel)
    plt.title("Expected vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_aor_residuals(df):
    """
    Plots AoR residuals (Actual - Expected) as points.
    """
    residuals = df["AoR"] - df["Expected_AoR"]
    x = range(len(df))

    plt.figure()
    plt.scatter(x, residuals)
    
    plt.axhline(0)
    plt.xlabel("Run")
    plt.ylabel("Residual AoR (deg)")
    plt.title("AoR Residuals (Actual - Expected)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# PART 1 REPRODUCEABILITY CHECK
# --------------------------------------------------------------------

results_reproduceability = []

for label, filename in files.items():
    csv_file = os.path.join(base_dir_reproduceability, filename)
    angle, max_z = compute_angle_of_repose_part_significance(csv_file)
    results_reproduceability.append([label, angle, max_z])
    #print(f"{label}: AoR = {angle:.2f}°, Max Z (binned) = {max_z:.3f}")

df_repoduceability = pd.DataFrame(results_reproduceability, columns=[
    "Friction Specification",
    "Angle of Repose (deg)",
    "Max Z (binned)"
])

print("\n============== FINAL RESULTS ==============")
print(df_repoduceability)
print("===========================================\n")

# --------------------------------------------------------------------
# PART 1.a) Velocity check
# --------------------------------------------------------------------

"""
There are 2 moments that could be visualized
1) From the start of the simulation which is at time=0
2) After opening the gate, which happens at time=1.3 (pref 1.25 is recommended in that case)
"""

# Load the files
#start_time_simulation = 0
start_time_after_opening_gate = 1.25
df_velocity = load_velocity_files(files_velocity, start_time=start_time_after_opening_gate ,base_dir=base_dir_velocty)

"""Uncomment if you want to see the graph"""
#plot_velocity(df_velocity)


# --------------------------------------------------------------------
# PART 1.b) Timestep analysis plot
# --------------------------------------------------------------------
quadrant_timesteps = ["0.20 timestep", "0.40 timestep", "0.60 timestep", "0.80 timestep"]
single_timestep = "1.00 timestep"

# ---- Load 20–80% for combined plot ----
combined_data = {}
for t in quadrant_timesteps:
    file = os.path.join(DATA_DIR, files_timestep[t])
    times, ke = parse_file(file)
    combined_data[t] = (times, ke)

# ---- Generate 2x2 combined plot ----
#make_combined_plot(combined_data, "KE_20_40_60_80_combined.png")
print("Generated: KE_20_40_60_80_combined.png")

# ---- Load 100% for separate plot ----
times, ke = parse_file(os.path.join(DATA_DIR, files_timestep[single_timestep]))
print("Loaded 100% timestep:", len(times), "data points")
#make_plot(times, ke, single_timestep, "KE_100_timestep.png")
print("Generated: KE_100_timestep.png")

print("All plots generated successfully.")


# -----------------------------------------------------------
# PART 2) SCREENING DOE TO FIND THE 3 MOST IMPORTANT PARAMETERS
# -----------------------------------------------------------

"""
Computes the importance of the parameters in the model
"""

results_doe_initial = []

for label, folder in files_Significance.items():
    
    # folder = "Run_1", "Run_2", etc.
    csv_file = os.path.join(dir_doe_significance, folder)
    angle, max_z = compute_angle_of_repose_part_significance(csv_file, VISUALIZE=False)
    results_doe_initial.append([label, angle, max_z])
    #print(f"{label}: AoR = {angle:.2f}°, Max Z (binned) = {max_z:.3f}")

df_doe_screening = pd.DataFrame(results_doe_initial, columns=["Friction Specification",
                                    "Angle of Repose (deg)",
                                    "Max Z (binned)"])

print("\n============== FINAL RESULTS ==============")
print(df_doe_screening)
print("===========================================\n")

signs["AoR"] = df_doe_screening["Angle of Repose (deg)"].values
effect_strength = {}

for param in ["A", "B", "C", "D", "E", "F", "G"]:
    mean_plus  = signs.loc[signs[param] ==  1, "AoR"].mean()
    mean_minus = signs.loc[signs[param] == -1, "AoR"].mean()
    effect_strength[param] = abs(mean_plus - mean_minus)

effect_strength = pd.Series(effect_strength).sort_values(ascending=False)
print(effect_strength)


# --------------------------------------------------------------------
# PART 3 PARAMETER ANALYSIS
# --------------------------------------------------------------------

results_ranking = []

for label, folder in files_Statistics.items():
    
    # folder = "Run_1", "Run_2", etc.
    csv_file = os.path.join(dir_Final_Doe, folder)
    angle, max_z = compute_angle_of_repose_part_significance(csv_file)
    results_ranking.append([label, angle, max_z])
    #print(f"{label}: AoR = {angle:.2f}°, Max Z (binned) = {max_z:.3f}")

df_ranking = pd.DataFrame(results_ranking, columns=["Friction Specification",
                                    "Angle of Repose (deg)",
                                    "Max Z (binned)"])

print("\n============== FINAL RESULTS ==============")
print(df_ranking)
print("===========================================\n")

# --------------------------------------------------------------------
# PART 4 STATISTICAL ANALYSIS OF RESULTS
# --------------------------------------------------------------------

rows = []

for r in runs:
    A_lvl = r["A"]
    B_lvl = r["B"]
    C_lvl = r["C"]

    rows.append({
        "run": r["run"],

        # level indices
        "A_level": A_lvl,
        "B_level": B_lvl,
        "C_level": C_lvl,

        # actual parameter values
        "A_restitution_pp": param_values["A_restitution_pp"][A_lvl],
        "B_static_friction_pp": param_values["B_static_friction_pp"][B_lvl],
        "C_rolling_friction_pp": param_values["C_rolling_friction_pp"][C_lvl],
    })

df_rows = pd.DataFrame(rows)
y_df = df_ranking[["Angle of Repose (deg)"]]

# 1. Merge output and input
df = df_rows.copy()
df['AoR'] = y_df['Angle of Repose (deg)']

print("\n============== DATAFRAME FOR STATISTICAL ANALYSIS ==============")
print(df.head())
print("==================================================================\n")

# --------------------------------------------------------------------
# PART 4.a) REGRESSION MODEL
# --------------------------------------------------------------------

# We use the 'pp' columns as they are the actual continuous physical values
model_cols = {
    'A': 'A_restitution_pp',
    'B': 'B_static_friction_pp',
    'C': 'C_rolling_friction_pp',
    'y': 'AoR'
}

# Define the formula: y ~ A + B + C + (C^(pow))
# The '*' operator in statsmodels automatically includes main effects and interactions
formula_basic = "AoR ~ A_restitution_pp + B_static_friction_pp + C_rolling_friction_pp"
formula_quadratic = "AoR ~ A_restitution_pp + B_static_friction_pp + C_rolling_friction_pp + I(C_rolling_friction_pp**2)"
formula_nonlinear = "AoR ~ A_restitution_pp + B_static_friction_pp + C_rolling_friction_pp + I(C_rolling_friction_pp**0.25)"
formula_interactions = "AoR ~ A_restitution_pp + B_static_friction_pp + C_rolling_friction_pp + I(A_restitution_pp*B_static_friction_pp) + I(A_restitution_pp*C_rolling_friction_pp) + I(B_static_friction_pp*C_rolling_friction_pp) "
formula_exponential = "AoR ~ A_restitution_pp + B_static_friction_pp + C_rolling_friction_pp + I(2.71**C_rolling_friction_pp)" 
formula_nonlinear_interaction = "AoR ~ A_restitution_pp + B_static_friction_pp + C_rolling_friction_pp + I(C_rolling_friction_pp**0.25) + I(B_static_friction_pp*C_rolling_friction_pp)"

model = smf.ols(formula=formula_nonlinear, data=df).fit()

# This displays the coefficients (Intercept, A, B, C, C**1.15)
print(model.summary())

df["Expected_AoR"] = model.predict(df)
df["Residual"] = df["AoR"] - df["Expected_AoR"]

#plot_expected_vs_actual(df,actual_col="AoR",expected_col="Expected_AoR")
#plot_aor_residuals(df)

# --------------------------------------------------------------------
# PART 4.b) MACHINE LEARNING MODEL
# --------------------------------------------------------------------

X = df[[
    "A_restitution_pp",
    "B_static_friction_pp",
    "C_rolling_friction_pp"
]]
y = df["AoR"]

# 22 train / 3 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=3,          # absolute number of test samples
    random_state=42
)

#Note   : If I do not add noise in the system, the model simply overfits the datam which is not desireable
#Note 2 : Does the graph I get with the noise come from the noise itself, or does the ML actually get better at it? 
    # No it does not, the WhiteKernel allows the moted to not overfit the data and it indeed produces an okay result
kernel = (
    C(1.0, (1e-3, 1e3))
    * RBF(length_scale=[1, 1, 1], length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 5))
)

gpr = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr", GaussianProcessRegressor(
        kernel=kernel,  
        normalize_y=True,
        n_restarts_optimizer=10
    ))
])

gpr.fit(X_train, y_train)
# Training predictions

y_train_pred = gpr.predict(X_train)

# Test predictions (this is what matters)
y_test_pred, y_test_std = gpr.predict(X_test, return_std=True)

df["Expected_AoR_GPR"] = np.nan

df.loc[X_train.index, "Expected_AoR_GPR"] = y_train_pred
df.loc[X_test.index, "Expected_AoR_GPR"] = y_test_pred

# Compute performance statistics
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
test_residuals = y_test - y_test_pred

print(f"Train RMSE: {rmse_train:.2f}, R²: {r2_train:.3f}")
print(f"Test  RMSE: {rmse_test:.2f}, R²: {r2_test:.3f}")
#print(test_residuals)

print()
print("===========================================\n")

gpr_model = gpr.named_steps["gpr"]
print(gpr_model.kernel_)
print("=======================\n")
length_scales = gpr_model.kernel_.k1.k2.length_scale

for name, ls in zip(X.columns, length_scales):
    print(f"{name}: length scale = {ls:.3f}")

signal_variance = gpr_model.kernel_.k1.k1.constant_value
print("=======================\n")
print(f"Signal variance: {signal_variance:.3f}")

noise_variance = gpr_model.kernel_.k2.noise_level
print(f"Noise variance: {noise_variance:.3f}")

importance = 1 / length_scales
importance /= importance.sum()

importance_df = pd.DataFrame({
    "Parameter": X.columns,
    "Length_scale": length_scales,
    "Relative_importance": importance
})
print("=======================\n")
print(importance_df)
print("===========================================\n")

#plot_expected_vs_actual(df,actual_col="AoR",expected_col="Expected_AoR_GPR")
#plot_expected_vs_actual(df,actual_col="AoR",expected_col="Expected_AoR_GPR",test_idx=X_test.index,train_color="blue",test_color="red",actual_color="yellow")


# --------------------------------------------------------------------
# PLOTTING REQUESTS
# --------------------------------------------------------------------

make_combined_plot(combined_data, "KE_20_40_60_80_combined.png")
make_plot(times, ke, single_timestep, "KE_100_timestep.png")
#plot_velocity(df_velocity)
#plot_expected_vs_actual(df,actual_col="AoR",expected_col="Expected_AoR")
#plot_aor_residuals(df)
#plot_expected_vs_actual(df,actual_col="AoR",expected_col="Expected_AoR_GPR")
#plot_expected_vs_actual(df,actual_col="AoR",expected_col="Expected_AoR_GPR",test_idx=X_test.index,train_color="blue",test_color="red",actual_color="yellow")