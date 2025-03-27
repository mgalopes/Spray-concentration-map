import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Set clean and bold style
plt.style.use("default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.figsize": (10, 8),
    "lines.linewidth": 2.5,
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

# Dictionaries for translations
fuel_translation = {"Etanol": "Ethanol", "Gasolina": "Gasoline"}
nozzle_translation = {"Divergente": "Divergent", "Convergente": "Convergent"}

# Define which fuels to plot
fuels_to_plot = ["Ethanol", "Water"]  # Modify this list to include/exclude fuels

def plot_from_xlsx(file_path):
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Replace Portuguese terms with English equivalents
    df['Fluid'] = df['Fluid'].replace(fuel_translation)
    df['Nozzle'] = df['Nozzle'].replace(nozzle_translation)
    
    # Filter to include only Divergent nozzles
    df = df[df['Nozzle'] == "Divergent"]
    
    # Compute global min and max for 'Ratio' and add a margin
    y_min = 0.5  # or use df["Ratio"].min() - 0.02
    y_max = 0.9  # or use df["Ratio"].max() + 0.02

    # Generate the legend entry
    df["Legend"] = df["Fluid"] + " " + df["Nozzle"] + " " + df["Temperature [ºC]"].astype(str)
    
    # Filter only selected fuels
    df = df[df["Fluid"].isin(fuels_to_plot)]
    
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

        # Set labels, title, and formatting
        ax.set_xlabel("Pressure [bar]", fontsize=18, fontweight="bold")
        ax.set_ylabel("Ratio", fontsize=18, fontweight="bold")
        ax.set_title(f"Experimental Data Plot - {fluid}", fontsize=22, fontweight="bold", pad=20)

        # Apply global y-axis limits
        ax.set_ylim(y_min, y_max)

        # Adjust ticks
        ax.tick_params(axis='both', labelsize=16)

        # Set x-axis ticks to every 10 bar
        ax.xaxis.set_major_locator(MultipleLocator(10))

        # Increase legend size
        ax.legend(fontsize=16, title="Legend", title_fontsize=18)

        # Add grid with transparency
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.show()

    # Combined plot for all selected fuels
    fig, ax = plt.subplots()
    for fluid in unique_fluids:
        df_fluid = df[df["Fluid"] == fluid]
        unique_legends = df_fluid["Legend"].unique()

        for legend in unique_legends:
            subset = df_fluid[df_fluid["Legend"] == legend]
            subset = subset.sort_values(by="Pressure [bar]")
            nozzle_type = subset["Nozzle"].iloc[0]
            marker_style = marker_dict.get(nozzle_type, 'o')
            ax.plot(subset["Pressure [bar]"], subset["Ratio"], marker=marker_style, linestyle='--', label=legend)

    # Set labels, title, and formatting for the combined plot
    ax.set_xlabel("Pressure [bar]", fontsize=18, fontweight="bold")
    ax.set_ylabel("Ratio", fontsize=18, fontweight="bold")
    ax.set_title("Combined Experimental Data Plot", fontsize=22, fontweight="bold", pad=20)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', labelsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(fontsize=16, title="Legend", title_fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    plot_from_xlsx("00_fixed_and_variable_areas.xlsx")

