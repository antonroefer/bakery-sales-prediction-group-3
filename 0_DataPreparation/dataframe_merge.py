import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data into DataFrames
umsatz = pd.read_csv("./Internal/umsatzdaten_gekuerzt.csv")
kiwo = pd.read_csv("./Internal/kiwo.csv")
wetter = pd.read_csv("./Internal/wetter.csv")

# Reshape 'umsatz' so that no duplicate 'Datum' exists
umsatz = umsatz.pivot_table(
    index="Datum", columns="Warengruppe", values="Umsatz", aggfunc="sum"
).reset_index()

# Rename columns to match the desired format
umsatz.columns = ["Datum"] + [f"Umsatz_WG_{i}" for i in range(1, len(umsatz.columns))]
# Merge the DataFrames on the 'Datum' column
merged_df = umsatz.merge(kiwo, on="Datum", how="left").merge(
    wetter, on="Datum", how="left"
)

# Fill NaN values with 0
merged_df.fillna(0, inplace=True)

# Convert 'Datum' to datetime and extract the weekday
merged_df["Datum"] = pd.to_datetime(merged_df["Datum"])
merged_df["Wochentag"] = merged_df["Datum"].dt.day_name()

# List of Umsatz columns
umsatz_columns = [col for col in merged_df.columns if col.startswith("Umsatz_WG_")]

# Create plots for each Warengruppe
for col in umsatz_columns:
    plt.figure(figsize=(10, 6))

    # Group by weekday and calculate mean and confidence interval
    weekday_stats = merged_df.groupby("Wochentag")[col].agg(["mean", "std", "count"])
    weekday_stats["ci"] = 1.96 * (
        weekday_stats["std"] / np.sqrt(weekday_stats["count"])
    )

    # Sort by weekday order
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_stats = weekday_stats.reindex(weekday_order)

    plt.bar(
        weekday_stats.index,
        weekday_stats["mean"],
        color="teal",
        yerr=weekday_stats["ci"],
        capsize=5,
    )
    plt.title(f"{col} nach Wochentag")
    plt.xlabel("Wochentag")
    plt.ylabel("Durchschnittlicher Umsatz")
    plt.xticks(rotation=45)
    plt.tight_layout()
