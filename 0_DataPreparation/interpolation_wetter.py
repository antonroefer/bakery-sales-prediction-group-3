import numpy as np
import pandas as pd
import base_df as base
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from collections import defaultdict

# set pandas options for better display
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# crate new dataframe for each year
wetter = base.wetter.copy()

wetter["Datum"] = pd.to_datetime(wetter["Datum"])

wetter["Jahr"] = wetter["Datum"].dt.year
wetter.set_index("Datum", inplace=True)

# check for missing dates
full_index = pd.date_range(start=wetter.index.min(), end=wetter.index.max(), freq="D")
missing_dates = full_index.difference(wetter.index)


# Dictionary: Jahr → DataFrame mit Daten dieses Jahres
jahres_daten = {jahr: gruppe for jahr, gruppe in wetter.groupby("Jahr")}

for jahr, df in jahres_daten.items():
    # Vollständigen Index mit Tagesfrequenz von min bis max im jeweiligen Jahr erzeugen
    voller_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")

    # Fehlende Tage mit NaN ergänzen
    df = df.reindex(voller_index)
    df.index.name = "Datum"

    jahres_daten[jahr] = df


# procedure: Savitzky-Golay-Filter


for jahr, df in jahres_daten.items():
    temperatur = df["Temperatur"].to_numpy()

    # Fehlende Werte zuerst mit linearer Interpolation auffüllen (damit Filter funktioniert)
    temperatur_interpoliert = (
        pd.Series(temperatur)
        .interpolate(method="linear", limit_direction="both")
        .to_numpy()
    )

    # Savitzky-Golay-Filter anwenden: Fenstergröße (z.B. 50) und Polynomgrad (z.B. 3)
    temperatur_geglättet = savgol_filter(
        temperatur_interpoliert, window_length=200, polyorder=3
    )

    # Geglättete Werte im DataFrame speichern
    df["Temperatur_Glatt"] = temperatur_geglättet

    jahres_daten[jahr] = df


# Für jedes Jahr Plot erzeugen und als PNG speichern
for jahr, df in jahres_daten.items():
    if "Temperatur_Glatt" not in df.columns:
        print(f"Jahr {jahr} wurde noch nicht geglättet – übersprungen.")
        continue

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["Temperatur"], label="Original", alpha=0.5)
    plt.plot(df.index, df["Temperatur_Glatt"], label="Geglättet", color="red")

    plt.title(f"Temperaturverlauf {jahr}")
    plt.xlabel("Datum")
    plt.ylabel("Temperatur (°C)")
    plt.legend()
    plt.tight_layout()

    # Speichern im Ordner "Plots"
    dateiname = f"Plots/temperaturverlauf_{jahr}.png"
    plt.savefig(dateiname, dpi=150)
    plt.close()  # Schließt das Plotfenster, damit Speicher nicht überläuft


# Dictionary: (Monat, Tag) → Liste von geglätteten Temperaturwerten
# Create a dictionary that maps each (month, day) pair to a list of smoothed temperatures
# Example: (1, 1) → [2.3, 1.8, 2.0, ...] for Jan 1 across different years
temperatur_je_datum = defaultdict(list)

# Loop through each year's DataFrame
for jahr, df in jahres_daten.items():
    # Skip this year if it hasn't been smoothed
    if "Temperatur_Glatt" not in df.columns:
        continue

    df = df.copy()
    # Reset index to access "Datum" as a regular column
    df = df.reset_index()
    df["Monat"] = df["Datum"].dt.month
    df["Tag"] = df["Datum"].dt.day

    # Loop through each row to group values by (month, day)
    for _, row in df.iterrows():
        schlüssel = (row["Monat"], row["Tag"])
        wert = row["Temperatur_Glatt"]

        # Only store non-missing (non-NaN) values
        if pd.notna(wert):
            temperatur_je_datum[schlüssel].append(wert)

# Mittelwerte berechnen
mittelwerte = {
    key: np.mean(werte) for key, werte in temperatur_je_datum.items() if werte
}

# In DataFrame überführen
durchschnitt_df = pd.DataFrame(
    {
        "Monat": [k[0] for k in mittelwerte.keys()],
        "Tag": [k[1] for k in mittelwerte.keys()],
        "Temperatur_Durchschnitt": list(mittelwerte.values()),
    }
)

# Für sinnvolle Sortierung
durchschnitt_df["Datum_sort"] = pd.to_datetime(
    {"year": 2000, "month": durchschnitt_df["Monat"], "day": durchschnitt_df["Tag"]}
)
durchschnitt_df.sort_values("Datum_sort", inplace=True)
plt.figure(figsize=(12, 4))
plt.plot(durchschnitt_df["Datum_sort"], durchschnitt_df["Temperatur_Durchschnitt"])
plt.title("Mittlerer geglätteter Temperaturverlauf (Monat/Tag-basiert)")
plt.xlabel("Datum im Jahr")
plt.ylabel("Temperatur (°C)")
plt.tight_layout()
plt.savefig("Plots/temperatur_mittelwert_alle_jahre_monat_tag.png", dpi=150)
plt.show()
