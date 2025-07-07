import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError as mean_squared_error
import tensorflow as tf

# Load the trained model
# model = load_model("python_model_3.h5", custom_objects={"mse": mean_squared_error})
model = load_model("python_model_N_OB_2.keras", compile=False)

# Define the file paths
subdirectory = "pickle_data"
training_features_path = f"{subdirectory}/training_features.pkl"
validation_features_path = f"{subdirectory}/validation_features.pkl"
test_features_path = f"{subdirectory}/test_features.pkl"
training_labels_path = f"{subdirectory}/training_labels.pkl"
validation_labels_path = f"{subdirectory}/validation_labels.pkl"
test_labels_path = f"{subdirectory}/test_labels.pkl"

# Read the pickle files
training_features = pd.read_pickle(training_features_path)
validation_features = pd.read_pickle(validation_features_path)
test_features = pd.read_pickle(test_features_path)
training_labels = pd.read_pickle(training_labels_path)
validation_labels = pd.read_pickle(validation_labels_path)
test_labels = pd.read_pickle(test_labels_path)


# Example: Predict on test_features
# validation_labels["Umsatz"] = model.predict(validation_features)

# test_labels[["id", "Umsatz"]].to_csv("nn_submission.csv", index=False)


# Calculate MAPE for each product group (Warengruppe_1 to Warengruppe_6)
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return (
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            )
        )
        * 100
    )


val_predictions = model.predict(validation_features).flatten()

# Calculate and store MAPE per Warengruppe (one-hot encoded columns Warengruppe_1 to Warengruppe_6)
mape_per_group = {}

with open("validation_mape_per_warengruppe.txt", "w") as f:
    for i in range(1, 7):
        col = f"Warengruppe_{i}"
        group_idx = validation_features[col] == 1
        group_true = validation_labels[group_idx]
        group_pred = val_predictions[group_idx]
        mape_metric = tf.keras.metrics.MeanAbsolutePercentageError()
        mape_metric.update_state(group_true, group_pred)
        group_mape = mape_metric.result().numpy()
        mape_per_group[col] = group_mape
        f.write(f"{col}: {group_mape:.4f}%\n")
