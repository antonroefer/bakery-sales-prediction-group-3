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
        BatchNormalization(),
        Dense(16, activation="sigmoid", kernel_regularizer=l2(1e-3)),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-3)),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
        Dense(1),
    ]
)

model.summary()

# Parameter
initial_learning_rate = 0.00015
decay_steps = 1000
decay_rate = 0.99
min_lr = 1e-5


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

early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)

history = model.fit(
    training_features,
    training_labels,
    epochs=1000,
    validation_data=(validation_features, validation_labels),
    callbacks=[early_stop],
)

model.save("python_model.keras")

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
