import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set scientific-style formatting
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (7, 5),
    "lines.linewidth": 2,
    "grid.linestyle": "--",
})

def plot_from_xlsx(file_path):
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Generate the legend entry
    df["Legend"] = df["Fluid"] + " " + df["Nozzle"] + " " + df["Temperature [ºC]"].astype(str)
    
    # Get unique fluids
    unique_fluids = df["Fluid"].unique()
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']
    nozzle_types = df["Nozzle"].unique()
    marker_dict = {nozzle: markers[i % len(markers)] for i, nozzle in enumerate(nozzle_types)}
    
    for fluid in unique_fluids:
        df_fluid = df[df["Fluid"] == fluid]
        unique_legends = df_fluid["Legend"].unique()
        
        # Create the plot
        fig, ax = plt.subplots()
        
        for legend in unique_legends:
            subset = df_fluid[df_fluid["Legend"] == legend]
            subset = subset.sort_values(by="Pressure [bar]")
            
            nozzle_type = subset["Nozzle"].iloc[0]
            marker_style = marker_dict.get(nozzle_type, 'o')
            
            ax.plot(subset["Pressure [bar]"], subset["Ratio"], marker=marker_style, linestyle='--', label=legend)
        
        ax.set_xlabel("Pressure [bar]")
        ax.set_ylabel("Ratio")
        ax.set_title(f"Experimental Data Plot - {fluid}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.show()

if __name__ == "__main__":
    plot_from_xlsx("00_fixed_and_variable_areas.xlsx")

