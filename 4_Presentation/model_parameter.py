import tensorflow as tf
from tensorflow import keras

# Angenommen, dein Modell ist bereits gespeichert unter 'mein_modell.keras'
model_filename = "../3_Model/python_model_N_OB_2.keras"

# --- 1. Modell laden ---
try:
    loaded_model = keras.models.load_model(model_filename)
    print(f"Modell erfolgreich von '{model_filename}' geladen.\n")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    print(
        f"Bitte stelle sicher, dass die Datei {model_filename} existiert und korrekt ist."
    )
    exit()

# --- 2. Informationen ausgeben ---

print("--- Informationen des geladenen Modells ---")

# A. Modellkonfiguration (Architektur)
print("\nA. Modellzusammenfassung (Architektur):")
loaded_model.summary()
print("\n")

# B. Optimierer-Konfiguration und Zustand
print("B. Optimierer-Konfiguration und Zustand:")
if loaded_model.optimizer is not None:
    print(f"  Optimierer-Klasse: {type(loaded_model.optimizer).__name__}")

    optimizer_config = loaded_model.optimizer.get_config()
    print("  Optimierer-Konfiguration:")
    for key, value in optimizer_config.items():
        print(f"    {key}: {value}")

    if hasattr(loaded_model.optimizer, "learning_rate"):
        lr_value = loaded_model.optimizer.learning_rate
        if isinstance(lr_value, tf.keras.optimizers.schedules.LearningRateSchedule):
            print(f"    Learning Rate (als Scheduler): {lr_value}")
        else:
            print(
                f"    Aktuelle Learning Rate (aus Optimierer-Objekt): {loaded_model.optimizer.learning_rate.numpy()}"
            )
    else:
        print("    Learning Rate nicht direkt als 'learning_rate' Attribut verfügbar.")

    print(
        "\n  Hinweis: Die internen Variablen des Optimierers (z.B. Momente) sind Teil seines Zustands,"
    )
    print("  aber nicht direkt über get_config() ausgebbar. Sie werden intern geladen.")

else:
    print("  Das geladene Modell wurde nicht mit einem Optimierer kompiliert.")
print("\n")

# C. Modellgewichte
print("C. Modellgewichte:")
print(
    f"  Anzahl der Schichten mit trainierbaren Gewichten: {len(loaded_model.weights)}"
)
if loaded_model.weights:
    print("  Beispielhafte Gewichte der ersten trainierbaren Schicht:")
    # Versuche, die Gewichte der ersten Schicht zu finden, die tatsächlich Gewichte hat
    found_weights = False
    for i, weights_tensor in enumerate(loaded_model.weights):
        if weights_tensor.shape.rank > 1:  # Typischerweise Gewichtsmatrizen
            print(
                f"    Gewichte von Schicht {i} (Form: {weights_tensor.shape}, Auszug):\n{weights_tensor.numpy()[:5, :5]}..."
            )
            found_weights = True
            break
    if not found_weights:
        print("  Keine geeigneten Gewichtsmatrizen für die Beispielausgabe gefunden.")

    print("  Beispielhafter Bias der ersten trainierbaren Schicht (falls vorhanden):")
    found_bias = False
    for i, weights_tensor in enumerate(loaded_model.weights):
        if weights_tensor.shape.rank == 1:  # Typischerweise Bias-Vektoren
            print(
                f"    Bias von Schicht {i} (Form: {weights_tensor.shape}, Auszug):\n{weights_tensor.numpy()[:5]}..."
            )
            found_bias = True
            break
    if not found_bias:
        print("  Keine geeigneten Bias-Vektoren für die Beispielausgabe gefunden.")
else:
    print("  Das Modell hat keine trainierbaren Gewichte oder ist noch nicht gebaut.")
print("\n")

# D. Kompilierungsinformationen (Loss und Metriken)
print("D. Kompilierungsinformationen:")
if loaded_model._is_compiled:
    print(f"  Loss-Funktion: {loaded_model.loss}")
    print(f"  Metriken: {loaded_model.metrics_names}")
else:
    print(
        "  Das geladene Modell wurde nicht kompiliert (oder die Kompilierungsinfos sind nicht verfügbar)."
    )
print("\n")
