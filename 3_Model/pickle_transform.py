import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./../2_BaselineModel/data_after_imputation.csv")

training_df = data[(data["Datum"] >= "2013-07-01") & (data["Datum"] <= "2017-07-31")]
validation_df = data[(data["Datum"] >= "2017-08-01") & (data["Datum"] <= "2018-07-31")]
test_df = data[(data["Datum"] >= "2018-08-01") & (data["Datum"] <= "2019-07-31")]

print("\nTraining set shape:", training_df.shape)
print("Validation set shape:", validation_df.shape)
print("Test set shape:", test_df.shape)

# Separating features and labels
training_features = training_df.drop(["id", "Datum", "Umsatz"], axis=1)
validation_features = validation_df.drop(["id", "Datum", "Umsatz"], axis=1)
test_features = test_df.drop(["id", "Datum", "Umsatz"], axis=1)

training_labels = training_df[["Umsatz"]]
validation_labels = validation_df[["Umsatz"]]
test_labels = test_df[["id", "Umsatz"]]

drop_features = False  # Set this to True to drop the columns

if drop_features:
    droppable_columns = [
        "Bewoelkung",
        "Windgeschwindigkeit",
        "Wahltag",
        "Niederschlag",
        "mask_Temperatur_Windgeschwindigkeit",
        "mask_Bewoelkung",
    ]

    training_features = training_features.drop(columns=droppable_columns)
    validation_features = validation_features.drop(columns=droppable_columns)
    test_features = test_features.drop(columns=droppable_columns)

    scaler = MinMaxScaler()

    # Fit scaler on training features and transform all sets
    training_features_scaled = pd.DataFrame(
        scaler.fit_transform(training_features),
        columns=training_features.columns,
        index=training_features.index,
    )
    validation_features_scaled = pd.DataFrame(
        scaler.transform(validation_features),
        columns=validation_features.columns,
        index=validation_features.index,
    )
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=test_features.columns,
        index=test_features.index,
    )

    # Replace original features with scaled versions
    training_features = training_features_scaled
    validation_features = validation_features_scaled
    test_features = test_features_scaled

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
