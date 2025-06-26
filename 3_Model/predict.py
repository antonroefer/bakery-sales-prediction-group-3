import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError as mean_squared_error

# Load the trained model
# model = load_model("python_model_3.h5", custom_objects={"mse": mean_squared_error})
model = load_model("python_model_A_3.keras", compile=False)

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
test_labels["Umsatz"] = model.predict(test_features)

test_labels[["id", "Umsatz"]].to_csv("nn_submission.csv", index=False)
