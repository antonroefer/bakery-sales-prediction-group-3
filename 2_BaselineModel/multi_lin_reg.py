import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file from the 0_DataPreparation directory
data = pd.read_csv("../0_DataPreparation/data.csv")

# Convert 'Datum' column to datetime and filter data up to 31.07.2018
data["Datum"] = pd.to_datetime(data["Datum"])
data = data[data["Datum"] <= "2018-07-31"]

# One-hot encode 'Warengruppe'
data = pd.get_dummies(data, columns=["Warengruppe"], drop_first=False, dtype=int)

print(data.head())

# Drop rows where 'Temperatur', 'Bewoelkung', or 'Windgeschwindigkeit' is NaN
data = data.dropna(subset=["Temperatur", "Bewoelkung", "Windgeschwindigkeit"])
for col in data.columns:
    print(f"{col}: {data[col].isna().sum()} NaNs")

# Print all unique dates where Umsatz > 1000
high_umsatz_dates = data.loc[data["Umsatz"] > 1000, "Datum"].drop_duplicates()
print("Unique dates with Umsatz > 1000:")
print(high_umsatz_dates)

features = [
    "Wochenende",
    "Silvester",
    # "Bewoelkung",
    "KielerWoche",
    "Temperatur",
    # "Windgeschwindigkeit",
    "Schulferien",
    "Feiertage",
    "VPI",
    # "Wettercode",
    # "Wahltag",
    "Niederschlag",
    "Warengruppe_1",
    "Warengruppe_2",
    "Warengruppe_3",
    "Warengruppe_4",
    "Warengruppe_5",
    "Warengruppe_6",
]

# Create a column for Warengruppe marker (1-6)
data["Warengruppe_marker"] = (
    data[
        [
            "Warengruppe_1",
            "Warengruppe_2",
            "Warengruppe_3",
            "Warengruppe_4",
            "Warengruppe_5",
            "Warengruppe_6",
        ]
    ]
    .idxmax(axis=1)
    .str.extract("(\d)")
    .astype(int)
)

# Select features for pairplot (add/remove as needed)
pairplot_features = [
    "Umsatz",
    "Temperatur",
    "Niederschlag",
    "VPI",
    "Wochenende",
    "Silvester",
    "KielerWoche",
    "Schulferien",
    "Feiertage",
]

sns.pairplot(
    data[pairplot_features + ["Warengruppe_marker"]],
    x_vars=[
        "Temperatur",
        "VPI",
        "Silvester",
        "Feiertage",
        "KielerWoche",
    ],
    y_vars=["Umsatz"],
    hue="Warengruppe_marker",
    palette="tab10",
    plot_kws={"alpha": 0.7},
)
plt.show()

Y = data["Umsatz"]
X = sm.add_constant(data[features])  # Add a constant term for the intercept
model = sm.OLS(Y, X)  # Ordinary Least Squares regression
results = model.fit()  # Fit the model

# Print the summary of the regression results
print(results.summary())
