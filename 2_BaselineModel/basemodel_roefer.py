import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file from the 0_DataPreparation directory
data_full = pd.read_csv("./data_after_imputation.csv")

# Convert 'Datum' column to datetime and filter data up to 31.07.2018
data_full["Datum"] = pd.to_datetime(data_full["Datum"])
data = data_full[data_full["Datum"] <= "2018-07-31"]
test = data_full[data_full["Datum"] > "2018-07-31"]


features = [
    "Wochenende",
    "Silvester",
    "Bewoelkung",
    "Temperatur",
    "Windgeschwindigkeit",
    "KielerWoche",
    "Schulferien",
    "Feiertage",
    "Wahltag",
    "VPI",
    "Niederschlag",
    "mask_Temperatur_Windgeschwindigkeit",
    "mask_Bewoelkung",
    "Wettercode_0",
    "Wettercode_10",
    "Wettercode_21",
    "Wettercode_5",
    "Wettercode_61",
    "Wettercode_63",
    "Wettercode_nan",
    "Wettercode_rest",
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
    data[features + ["Warengruppe_marker"] + ["Umsatz"]],
    x_vars=[
        "Temperatur",
        "VPI",
        "Silvester",
        "Feiertage",
        "KielerWoche",
        "Wettercode_0",
        "Wettercode_10",
        "Wettercode_21",
        "Wettercode_5",
        "Wettercode_61",
        "Wettercode_63",
        "Wettercode_nan",
        "Wettercode_rest",
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

# Predict Umsatz for the test dataset
X_test = sm.add_constant(test[features])
test["Umsatz"] = results.predict(X_test)

# Optionally, display the first few predictions
print(test.head())

# Export 'id' and predicted 'Umsatz' for the test set as submission.csv
submission = test[["id", "Umsatz"]]
submission.to_csv("submission_all.csv", index=False)
