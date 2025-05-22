import pandas as pd
from sklearn.linear_model import LinearRegression

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

training_df = training_df[["Temperatur", "Umsatz"]].dropna()

X = training_df[["Temperatur"]]  # ✅ 2D
y = training_df["Umsatz"]  # ✅ 1D

modell = LinearRegression()
modell.fit(X, y)

print("Steigung (Slope):", modell.coef_[0])
print("Achsenabschnitt (Intercept):", modell.intercept_)
print("Bestimmtheitsmaß R²:", modell.score(X, y))
