import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

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

model = Sequential(
    [
        InputLayer(shape=(training_features.shape[1],)),
        Dense(16, activation="sigmoid", kernel_regularizer=l2(1e-3)),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-3)),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
        Dense(1),
    ]
)

# Parameter
initial_learning_rate = 30e-5
decay_steps = 1000
decay_rate = 0.99
min_lr = 5e-6


# ExponentialDecay mit Minimum-Schranke
class ExponentialDecayWithMin(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, initial_learning_rate, decay_steps, decay_rate, min_lr, staircase=True
    ):
        self.decay_schedule = ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )
        self.min_lr = min_lr

    def __call__(self, step):
        return tf.maximum(self.decay_schedule(step), self.min_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.decay_schedule.initial_learning_rate,
            "decay_steps": self.decay_schedule.decay_steps,
            "decay_rate": self.decay_schedule.decay_rate,
            "min_lr": self.min_lr,
            "staircase": self.decay_schedule.staircase,
        }


# Scheduler-Instanz
lr_schedule = ExponentialDecayWithMin(
    initial_learning_rate, decay_steps, decay_rate, min_lr, staircase=True
)
# Optimizer mit Scheduler
optimizer = Adam(learning_rate=lr_schedule)

# Model kompilieren
model.compile(loss="mse", optimizer=optimizer)
early_stop = EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
# Combine training and validation data
# combined_features = pd.concat(
#     [training_features, validation_features], ignore_index=True
# )
# combined_labels = pd.concat([training_labels, validation_labels], ignore_index=True)

history = model.fit(
    training_features,
    training_labels,
    epochs=1000,
    validation_data=(validation_features, validation_labels),
    callbacks=[early_stop],
)

model.save("python_model.keras")

# Print the best (minimum) training and validation loss
best_epoch = history.history["val_loss"].index(min(history.history["val_loss"]))
best_train_loss = history.history["loss"][best_epoch]
best_val_loss = history.history["val_loss"][best_epoch]
print(f"Best epoch: {best_epoch}")
print(f"Best Training Loss (MSE): {best_train_loss}")
print(f"Best Validation Loss (MSE): {best_val_loss}")

# Predict on validation set and calculate MAPE
val_predictions = model.predict(validation_features).flatten()

# Correctly calculate MAPE
# 1. Create an instance of the metric
mape_metric = tf.keras.metrics.MeanAbsolutePercentageError()
# 2. Update the state with the true labels and predictions
mape_metric.update_state(validation_labels, val_predictions)
# 3. Get the result from the metric
val_mape = mape_metric.result().numpy()

print(f"Validation MAPE: {val_mape:.4f}%")

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

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
