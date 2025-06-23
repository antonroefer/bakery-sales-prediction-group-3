import pandas as pd
import os

# Load the CSV files
train = pd.read_csv("Internal/train.csv")
test = pd.read_csv("Internal/test.csv")

# Concatenate train and test DataFrames
data = pd.concat([train, test], ignore_index=True)

# Sort by 'Datum' first, then by 'Warengruppe'
data = data.sort_values(by=["Datum", "Warengruppe"]).reset_index(drop=True)

# Create 'Wochenende' feature: 1 if Saturday or Sunday, else 0
data["Wochenende"] = pd.to_datetime(data["Datum"]).dt.weekday.isin([5, 6]).astype(int)

# Create 'Silvester' column: 1 if 'Datum' is December 31, else 0
data["Silvester"] = (
    pd.to_datetime(data["Datum"]).dt.strftime("%m-%d").eq("12-31").astype(int)
)

# Load wetter.csv and kiwo.csv
wetter = pd.read_csv("Internal/wetter.csv")
kiwo = pd.read_csv("Internal/kiwo.csv")

# Merge wetter.csv on 'Datum'
data = pd.merge(data, wetter, on="Datum", how="left")

# Merge kiwo.csv on 'Datum'
data = pd.merge(data, kiwo, on="Datum", how="left")
data["KielerWoche"] = data["KielerWoche"].fillna(0)

# Load Schulferien.csv
schulferien = pd.read_csv("External/Schulferien.csv")
data = pd.merge(data, schulferien, on="Datum", how="left")

# Load Feiertage_SH.csv
feiertage = pd.read_csv("External/Feiertage_SH.csv")
feiertage["Datum"] = pd.to_datetime(feiertage["Datum"], dayfirst=True).dt.strftime(
    "%Y-%m-%d"
)
data["Feiertage"] = data["Datum"].isin(feiertage["Datum"]).astype(int)

# Load wahltage_binaer.csv
wahltage = pd.read_csv("External/wahltage_binaer.csv")
wahltage["Datum"] = pd.to_datetime(wahltage["Datum"]).dt.strftime("%Y-%m-%d")
data = pd.merge(data, wahltage, on="Datum", how="left")
data = data.rename(columns={"NeueSpalte": "Wahltag"})
data["Wahltag"] = data["Wahltag"].fillna(0)

# Load VPI_modified.csv and merge VPI
vpi = pd.read_csv("External/VPI_modified.csv")
vpi["Datum"] = pd.to_datetime(vpi["Datum"]).dt.strftime("%Y-%m-%d")
data = pd.merge(data, vpi[["Datum", "VPI"]], on="Datum", how="left")

# Load precipitation data
precipitation = pd.read_csv("External/precipitation.txt", sep=",")
precipitation["Datum"] = pd.to_datetime(
    precipitation["Datum"], format="%Y-%m-%d"
).dt.strftime("%Y-%m-%d")
# Create 'prec_categorie' column based on precipitation boundaries
bins = [-float("inf"), 2.5, 7.5, 36, 65, float("inf")]
labels = [0, 1, 2, 3, 4]
precipitation["Niederschlag"] = pd.cut(
    precipitation["Precipitation_mm"], bins=bins, labels=labels, right=True
).astype(int)
precipitation = precipitation.drop(columns=["Precipitation_mm"])
data = pd.merge(data, precipitation, on="Datum", how="left")

data.to_csv("data.csv", index=False)

training_df = data[(data["Datum"] >= "2013-07-01") & (data["Datum"] <= "2017-07-31")]
validation_df = data[(data["Datum"] >= "2017-08-01") & (data["Datum"] <= "2018-07-31")]
test_df = data[(data["Datum"] >= "2018-08-01") & (data["Datum"] <= "2019-07-31")]

print("\nTraining set shape:", training_df.shape)
print("Validation set shape:", validation_df.shape)
print("Test set shape:", test_df.shape)

# Separating features and labels
training_features = training_df.drop("price", axis=1)
validation_features = validation_df.drop("price", axis=1)
test_features = test_df.drop("price", axis=1)

training_labels = training_df[["price"]]
validation_labels = validation_df[["price"]]
test_labels = test_df[["price"]]

# Print dimensions of the dataframes
print("Training features dimensions:", training_features.shape)
print("Validation features dimensions:", validation_features.shape)
print("Test features dimensions:", test_features.shape)
print()
print("Training labels dimensions:", training_labels.shape)
print("Validation labels dimensions:", validation_labels.shape)
print("Test labels dimensions:", test_labels.shape)

# Create subdirectory for the pickle files
subdirectory = "pickle_data"
os.makedirs(subdirectory, exist_ok=True)

# Export of the prepared data to subdirectory as pickle files
training_features.to_pickle(f"{subdirectory}/training_features.pkl")
validation_features.to_pickle(f"{subdirectory}/validation_features.pkl")
test_features.to_pickle(f"{subdirectory}/test_features.pkl")
training_labels.to_pickle(f"{subdirectory}/training_labels.pkl")
validation_labels.to_pickle(f"{subdirectory}/validation_labels.pkl")
test_labels.to_pickle(f"{subdirectory}/test_labels.pkl")
