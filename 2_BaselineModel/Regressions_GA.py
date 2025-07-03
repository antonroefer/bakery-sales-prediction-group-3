# Here we want to perform linear and multiple linear regression on the data
# Goal is to predict the Umsatz based on the other features and find the best predictor(s)

# Run linear regressions on the training set only 01.07.2013 to 31.07.2017.
# Use the validation set to check model’s performance.
# For example, after fitting a model, predict Umsatz on the validation set (data from 01.08.2017 to 31.07.2018).
# Calculate metrics like RMSE or R² to see how well your model generalizes.
# test it on the test set (from 01.08.2018 to 31.07.2019)

# Libraries
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Data input
# Load the CSV file from the 0_DataPreparation directory
data = pd.read_csv("../0_DataPreparation/data.csv")
data["Datum"] = pd.to_datetime(data["Datum"])
print(data.columns)
# Some data preprocessing
data = data.drop(columns=["Wettercode"])  # Drop the 'Wettercode' column entirely

# NAs in these cols  so drop
# data = data.dropna(
#    subset=["Temperatur", "Bewoelkung", "Windgeschwindigkeit"])


# One-hot encode 'Warengruppe' in the full dataset
data = pd.get_dummies(data, columns=["Warengruppe"], prefix="Warengruppe", dtype=int)

# copied from the 0_DataPreparation directory/dataset.py
training_df = data[(data["Datum"] >= "2013-07-01") & (data["Datum"] <= "2017-07-31")]
validation_df = data[(data["Datum"] >= "2017-08-01") & (data["Datum"] <= "2018-07-31")]
test_df = data[(data["Datum"] >= "2018-08-01") & (data["Datum"] <= "2019-07-31")]
print("\nTraining set shape:", training_df.shape)
print("Validation set shape:", validation_df.shape)
print("Test set shape:", test_df.shape)

# Drop NAs ONLY from training and validation sets
training_df = training_df.dropna(
    subset=["Temperatur", "Bewoelkung", "Windgeschwindigkeit"]
)
validation_df = validation_df.dropna(
    subset=["Temperatur", "Bewoelkung", "Windgeschwindigkeit"]
)

print("Missing values in training set:")
print(training_df.isna().sum())

print("\nMissing values in validation set:")
print(validation_df.isna().sum())

print("\nMissing values in test set:")
print(test_df.isna().sum())

# Correlation matrix on training data
corr = training_df.corr()
print("Correlations with Umsatz:")
print(corr["Umsatz"].sort_values(ascending=False))

# Optional: heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix (training data)")
plt.show()

# regression with some predictors (based on correlation output)
predictors = ["Temperatur"]

# Simple linear regressions on training data (one predictor at a time)
for predictor in predictors:
    X = sm.add_constant(training_df[predictor])
    Y = training_df["Umsatz"]
    model = sm.OLS(Y, X).fit()
    # print(f"--- Model with predictor: {predictor} ---")
    # print(f"R-squared: {model.rsquared:.3f}")
    print(model.summary())
    print("\n")

# Select predictors based on correlations
predictors2 = [
    "Warengruppe_2",
    "Warengruppe_5",
    "Temperatur",
    "Schulferien",
    "Wochenende",
    "Silvester",
]

# Define X and Y for training set
X_train = training_df[predictors2]
Y_train = training_df["Umsatz"]

# Add constant (intercept)
X_train_const = sm.add_constant(X_train)

# Fit OLS regression model
model = sm.OLS(Y_train, X_train_const)
results = model.fit()

# Show summary including R-squared
print(results.summary())

# Check model performance on validation set


# Prepare validation predictors
X_val = validation_df[predictors2]
X_val_const = sm.add_constant(X_val)

# Predict Umsatz on validation data
Y_val_true = validation_df["Umsatz"]
Y_val_pred = results.predict(X_val_const)

# Calculate R² and RMSE
r2 = r2_score(Y_val_true, Y_val_pred)
rmse = np.sqrt(mean_squared_error(Y_val_true, Y_val_pred))

print(f"Validation R²: {r2:.3f}")
print(f"Validation RMSE: {rmse:.2f}")

# --- Predict on test set using the trained model ---
# Prepare test predictors (must match training columns)
X_test = test_df[predictors2]
X_test_const = sm.add_constant(X_test)

# Predict Umsatz
test_df["Umsatz_pred"] = results.predict(X_test_const)

# Prepare submission: only 'id' and 'Umsatz'
submission = test_df[["id", "Umsatz_pred"]].rename(columns={"Umsatz_pred": "Umsatz"})


print(submission.shape)  # Should be (1830, 2)
print(submission.head())

# # Save to CSV
# submission.to_csv("submission.csv", index=False)
# #print("✅ Submission file saved as submission.csv")


# #count nas in submission
# print("\nMissing values in submission:")
# print(submission.isna().sum())


# #see missing values in each col of wetter dataset from the internal
# missingdatacheck = pd.read_csv("../0_DataPreparation/data.csv")
# print("\nMissing values in original data:")
# print(missingdatacheck.isna().sum())

# #see dates of the days with missing values in the original data
# missing_dates = missingdatacheck[missingdatacheck.isna().any(axis=1)]["Datum"]
# print("\nDates with missing values in original data:")
# print(missing_dates.sort_values().unique())
