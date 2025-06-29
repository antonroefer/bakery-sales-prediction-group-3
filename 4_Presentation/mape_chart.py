import matplotlib.pyplot as plt
import os

# Daten
warengruppen = [
    "Warengruppe_1",
    "Warengruppe_2",
    "Warengruppe_3",
    "Warengruppe_4",
    "Warengruppe_5",
    "Warengruppe_6",
]
mape_scores = [25.5096, 14.6359, 20.7997, 26.8823, 17.0401, 51.8399]
mape_scores_2 = [23.7933, 12.6847, 19.5689, 22.8171, 14.9398, 62.1810]

# Bar Chart erstellen
plt.figure(figsize=(20, 12))
plt.rcParams.update({"font.size": 16})  # Schriftgrößen verdoppeln (Standard ist 12)
bars = plt.bar(warengruppen, mape_scores, color="skyblue")
plt.ylabel("MAPE (%)")
plt.xlabel("Warengruppe")
plt.title("MAPE Scores je Warengruppe")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.7)

# Werte auf den Balken anzeigen
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{yval:.2f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create the full path for the output file
output_path = os.path.join(script_dir, "mape_chart.png")

print(f"Attempting to save chart to: {output_path}")

plt.savefig(output_path, dpi=600)

if os.path.exists(output_path):
    print("Success! File has been saved.")
else:
    print(
        "Error: File was not created. Please check for errors in the terminal and verify folder permissions."
    )

plt.show()
