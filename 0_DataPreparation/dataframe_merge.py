import pandas as pd

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
