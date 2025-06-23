import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

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

initial_learning_rate = 0.0003  # or your starting learning rate
decay_steps = 1000  # number of steps before applying decay
decay_rate = 0.9  # decay factor (0 < decay_rate < 1)

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True,  # if True, decays discrete step-wise; if False, decays smoothly
)

model.compile(loss="mse", optimizer=Adam(learning_rate=1 / 3 * initial_learning_rate))

early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)

history = model.fit(
    training_features,
    training_labels,
    epochs=200,
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
