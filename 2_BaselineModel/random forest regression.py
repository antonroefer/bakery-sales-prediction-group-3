# Here we load the training features and labels from pickle files, 
#    fit a Random Forest model, and visualize the feature importances.
# We used that while working on the Neural Network, where the first part of the code
#    includes some features selection and then we wanted to see how important the features are.


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Define the file paths
subdirectory = "./../3_Model/pickle_data"
training_features_path = f"{subdirectory}/training_features.pkl"
training_labels_path = f"{subdirectory}/training_labels.pkl"

# Read the pickle files
training_features = pd.read_pickle(training_features_path)
training_labels = pd.read_pickle(training_labels_path)


# random forest regression

# Convert labels to 1D numpy array
y_train = training_labels.to_numpy().ravel()
X_train = training_features

# Fit Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = training_features.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
