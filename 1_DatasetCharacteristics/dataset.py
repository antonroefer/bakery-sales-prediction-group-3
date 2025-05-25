import pandas as pd

# Load the CSV files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Concatenate train and test DataFrames
data = pd.concat([train, test], ignore_index=True)

# Sort by 'Datum' first, then by 'Warengruppe'
data = data.sort_values(by=["Datum", "Warengruppe"]).reset_index(drop=True)

# Create 'Wochenende' feature: 1 if Saturday or Sunday, else 0
data["Wochenende"] = pd.to_datetime(data["Datum"]).dt.weekday.isin([5, 6]).astype(int)

# Load wetter.csv and kiwo.csv
wetter = pd.read_csv("../0_DataPreparation/Internal/wetter.csv")
kiwo = pd.read_csv("../0_DataPreparation/Internal/kiwo.csv")

# Merge wetter.csv on 'Datum'
data = pd.merge(data, wetter, on="Datum", how="left")

# Merge kiwo.csv on 'Datum'
data = pd.merge(data, kiwo, on="Datum", how="left")
data["KielerWoche"] = data["KielerWoche"].fillna(0)

# Load Schulferien.csv
schulferien = pd.read_csv("Schulferien.csv")
data = pd.merge(data, schulferien, on="Datum", how="left")

# Load Feiertage_SH.csv
feiertage = pd.read_csv("Feiertage_SH.csv")
feiertage["Datum"] = pd.to_datetime(feiertage["Datum"], dayfirst=True).dt.strftime(
    "%Y-%m-%d"
)
data["Feiertage"] = data["Datum"].isin(feiertage["Datum"]).astype(int)

# Load wahltage_binaer.csv
wahltage = pd.read_csv("wahltage_binaer.csv")
wahltage["Datum"] = pd.to_datetime(wahltage["Datum"]).dt.strftime("%Y-%m-%d")
data = pd.merge(data, wahltage, on="Datum", how="left")
data = data.rename(columns={"NeueSpalte": "Wahltag"})
data["Wahltag"] = data["Wahltag"].fillna(0)

# Load VPI_modified.csv and merge VPI
vpi = pd.read_csv("VPI_modified.csv")
vpi["Datum"] = pd.to_datetime(vpi["Datum"]).dt.strftime("%Y-%m-%d")
data = pd.merge(data, vpi[["Datum", "VPI"]], on="Datum", how="left")

data.to_csv("data.csv", index=False)
