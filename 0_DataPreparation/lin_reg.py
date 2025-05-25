import pandas as pd
from scipy.stats import linregress

# Sales data
umsatzdaten = pd.read_csv("Internal/umsatzdaten_gekuerzt.csv")

# Weather data
wetter = pd.read_csv("Internal/wetter.csv")

# Kieler Woche data
kiwo = pd.read_csv("Internal/kiwo.csv")

# Ensure date format is consistent
umsatzdaten["Datum"] = pd.to_datetime(umsatzdaten["Datum"])
kiwo["Datum"] = pd.to_datetime(kiwo["Datum"])
wetter["Datum"] = pd.to_datetime(wetter["Datum"])

# Merge the DataFrames on the 'Datum' column
merged_df = umsatzdaten.merge(kiwo, on="Datum", how="outer").merge(
    wetter, on="Datum", how="outer"
)

# Ensure 'Datum' to datetime and extract the year, day of the year, and weekday
merged_df["Datum"] = pd.to_datetime(merged_df["Datum"])

# Create a DataFrame with all combinations of Datum and Warengruppe (1-6)
all_dates = pd.DataFrame({"Datum": merged_df["Datum"].unique()})
all_warengruppen = pd.DataFrame({"Warengruppe": range(1, 7)})
all_combinations = all_dates.merge(all_warengruppen, how="cross")

# Merge with the original dataframe to ensure every date has all 6 Warengruppen
merged_df = all_combinations.merge(merged_df, on=["Datum", "Warengruppe"], how="left")

# Optional: sort by date and Warengruppe for readability
merged_df = merged_df.sort_values(["Datum", "Warengruppe"]).reset_index(drop=True)

# Constructing new variables for later use
merged_df["Jahr"] = merged_df["Datum"].dt.year
merged_df["Monat"] = merged_df["Datum"].dt.month
merged_df["Tag_im_Jahr"] = merged_df["Datum"].dt.dayofyear
merged_df["Wochentag"] = merged_df["Datum"].dt.weekday + 1  # 1=Monday, 7=Sunday

# in 'KielerWoche' fill NaN values  with 0
merged_df["KielerWoche"] = merged_df["KielerWoche"].fillna(
    0
)  # 0 = no Kieler Woche, 1 = Kieler Woche

# Set the ID in the format yymmddX (e.g., 1307053 for 2013-07-05, Warengruppe 3)
merged_df["id"] = merged_df["Datum"].dt.strftime("%y%m%d") + merged_df[
    "Warengruppe"
].astype(str)
merged_df["id"] = merged_df["id"].astype(int)


# Just for review ------------------------------------------------------------------


# Checking the data types
# print("\nData types:")
# print(merged_df.dtypes)

# print(merged_df["id"])

merged_df.to_html("merged_df.html", index=False)


training_df = merged_df[merged_df["Datum"] <= "2017-07-31"]
validation = merged_df[
    (merged_df["Datum"] > "2017-07-31") & (merged_df["Datum"] <= "2018-07-31")
]
test_df = merged_df[
    (merged_df["Datum"] > "2018-07-31") & (merged_df["Datum"] <= "2019-07-31")
]

# After creating test_df with the date filter
print(f"Initial test_df shape: {test_df.shape}")
print(f"Test period date range: {test_df['Datum'].min()} to {test_df['Datum'].max()}")
print(f"Count of non-null temperature values: {test_df['Temperatur'].count()}")

# Look at a few rows to check the data
print(test_df[["Datum", "Temperatur"]].head())

# If you find rows with NaN temperature, check why they didn't merge properly
if test_df["Temperatur"].isna().any():
    missing_temp = test_df[test_df["Temperatur"].isna()]
    print(f"Sample dates with missing temperature: {missing_temp['Datum'].head()}")

# Drop rows with missing values in 'Temperatur' or 'Umsatz'
train = training_df.dropna(subset=["Temperatur", "Umsatz"])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(
    train["Temperatur"], train["Umsatz"]
)

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")

# Predict 'Umsatz' for the test set using the linear regression formula
test_df = test_df.dropna(subset=["Temperatur"])
test_df["Umsatz_pred"] = intercept + slope * test_df["Temperatur"]

# Optionally, print or save the predictions
print(test_df[["Datum", "Warengruppe", "Temperatur", "Umsatz_pred"]].head())

# Export training data with 'id' and 'Umsatz' columns to CSV
test_df[["id", "Umsatz_pred"]].to_csv("test_id_umsatz.csv", index=False)
