# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import statsmodels.api as sm  # For statistical models, including OLS regression

# Load the full dataset from a CSV file located in a relative directory
data_full = pd.read_csv(
    "../1_DatasetCharacteristics/imputed_data/data_after_imputation_knn.csv"
)

# Convert the 'Datum' column from string to datetime objects for time-based filtering
data_full["Datum"] = pd.to_datetime(data_full["Datum"])

# Split the data into a training + validation set and a test set based on the date
# Training and Validation data includes all records up to and including 31.07.2018
data = data_full[data_full["Datum"] <= "2018-07-31"]
# Test data includes all records after 31.07.2018
test = data_full[data_full["Datum"] > "2018-07-31"]


# Define the list of feature columns to be used as independent variables in the model
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

# Prepare the data for the regression model
Y = data["Umsatz"]  # Define the dependent variable (target)
X = sm.add_constant(
    data[features]
)  # Define the independent variables (features) and add a constant for the intercept

# Create and fit the Ordinary Least Squares (OLS) regression model
model = sm.OLS(Y, X)  # Initialize the OLS model
results = model.fit()  # Fit the model to the data

# Print the detailed summary of the regression results
print(results.summary())

# Use the trained model to make predictions on the test set
X_test = sm.add_constant(test[features])  # Prepare the test features, adding a constant
test["Umsatz"] = results.predict(
    X_test
)  # Predict 'Umsatz' and store it in the test dataframe

# Display the first few rows of the test data with the new predictions
print(test.head())

# Prepare the final submission file
# Select the 'id' and predicted 'Umsatz' columns from the test set
submission = test[["id", "Umsatz"]]
# Export the submission dataframe to a CSV file without the pandas index
submission.to_csv("submission_all.csv", index=False)
