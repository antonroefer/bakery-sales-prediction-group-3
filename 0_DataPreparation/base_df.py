import pandas as pd

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

# Erstelle ein DataFrame mit allen Daten von 01.07.2013 bis 31.07.2019
dates = pd.date_range(start="2013-07-01", end="2019-07-31", freq="D")
df = pd.DataFrame({"datum": dates})
df["holiday"] = 0  # Standardwert: kein Feiertag

# Definition der Ferienzeiträume nach Schuljahren
ferien = [
    # (start_date, end_date)
    ("2012-10-04", "2012-10-19"),
    ("2012-12-24", "2013-01-05"),
    ("2013-03-25", "2013-04-09"),
    ("2013-05-10", "2013-05-10"),
    ("2013-06-24", "2013-08-03"),
    ("2013-10-04", "2013-10-18"),
    ("2013-12-23", "2014-01-06"),
    ("2014-04-16", "2014-05-02"),
    ("2014-05-30", "2014-05-30"),
    ("2014-07-14", "2014-08-23"),
    ("2014-10-13", "2014-10-25"),
    ("2014-12-22", "2015-01-06"),
    ("2015-04-01", "2015-04-17"),
    ("2015-05-15", "2015-05-15"),
    ("2015-07-20", "2015-08-29"),
    ("2015-10-19", "2015-10-31"),
    ("2015-12-21", "2016-01-06"),
    ("2016-03-24", "2016-04-09"),
    ("2016-05-06", "2016-05-06"),
    ("2016-07-25", "2016-09-03"),
    ("2016-10-17", "2016-10-29"),
    ("2016-12-23", "2017-01-06"),
    ("2017-04-07", "2017-04-21"),
    ("2017-05-26", "2017-05-26"),
    ("2017-07-24", "2017-09-02"),
    ("2017-10-16", "2017-10-27"),
    ("2017-12-21", "2018-01-06"),
    ("2018-03-29", "2018-04-13"),
    ("2018-05-11", "2018-05-11"),
    ("2018-07-09", "2018-08-18"),
    ("2018-10-01", "2018-10-19"),
    ("2018-12-21", "2019-01-04"),
    ("2019-04-04", "2019-04-18"),
    ("2019-05-31", "2019-05-31"),
    ("2019-07-01", "2019-08-10"),
]

# Setze holiday = 1 für alle Daten, die in einem der Ferienzeiträume liegen
for start, end in ferien:
    mask = (df["datum"] >= pd.to_datetime(start)) & (df["datum"] <= pd.to_datetime(end))
    df.loc[mask, "holiday"] = 1

    # Merge holiday information into merged_df, preserving all dates in merged_df
merged_df = merged_df.merge(
    df.rename(columns={"datum": "Datum"}), on="Datum", how="left"
)
merged_df["holiday"] = merged_df["holiday"].fillna(0).astype(int)

# Downcast all float and int columns to float32 and int32
for col in merged_df.select_dtypes(include=["float", "int"]).columns:
    if pd.api.types.is_float_dtype(merged_df[col]):
        merged_df[col] = merged_df[col].astype("float32")
    elif pd.api.types.is_integer_dtype(merged_df[col]):
        merged_df[col] = merged_df[col].astype("int32")

# Checking the data types
print("\nData types:")
print(merged_df.dtypes)

print(merged_df["id"])

merged_df.to_html("merged_df.html", index=False)

training_df = merged_df[
    (merged_df["Datum"] >= "2013-07-01") & (merged_df["Datum"] <= "2017-07-31")
]
validation_df = merged_df[
    (merged_df["Datum"] >= "2017-08-01") & (merged_df["Datum"] <= "2018-07-31")
]
test_df = merged_df[
    (merged_df["Datum"] >= "2018-08-01") & (merged_df["Datum"] <= "2019-07-31")
]

print("\nTraining set:")
print(training_df.shape)
print("\nValidation set:")
print(validation_df.shape)
print("\nTest set:")
print(test_df.shape)
