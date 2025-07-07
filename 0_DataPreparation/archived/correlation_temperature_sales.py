import numpy as np
import pandas as pd
import base_df as base
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr


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


for jahr, df in jahres_daten.items():
    # Vollständigen Index mit Tagesfrequenz von min bis max im jeweiligen Jahr erzeugen
    voller_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")

    # Fehlende Tage mit NaN ergänzen
    df = df.reindex(voller_index)
    df.index.name = "Datum"

    jahres_daten[jahr] = df


# procedure: Savitzky-Golay-Filter mit zwei Fenstergrößen


for jahr, df in jahres_daten.items():
    temperatur = df["Temperatur"].to_numpy()

    # Fehlende Werte zuerst mit linearer Interpolation auffüllen (damit Filter funktioniert)
    temperatur_interpoliert = (
        pd.Series(temperatur)
        .interpolate(method="linear", limit_direction="both")
        .to_numpy()
    )

    # Savitzky-Golay-Filter anwenden: Fenstergröße 200 und 11, Polynomgrad 3
    temperatur_geglättet_200 = savgol_filter(
        temperatur_interpoliert, window_length=200, polyorder=3
    )

    temperatur_geglättet_11 = savgol_filter(
        temperatur_interpoliert, window_length=11, polyorder=3
    )

    # Geglättete Werte im DataFrame speichern
    df["Temperatur_Glatt_200"] = temperatur_geglättet_200
    df["Temperatur_Glatt_11"] = temperatur_geglättet_11

    jahres_daten[jahr] = df


# Differenz zwischen Original- und geglätteter Temperatur berechnen
for jahr, df in jahres_daten.items():
    df["Temperatur_Differenz_200"] = df["Temperatur"] - df["Temperatur_Glatt_200"]
    df["Temperatur_Differenz_11"] = df["Temperatur"] - df["Temperatur_Glatt_11"]
    jahres_daten[jahr] = df


merged_df = base.merged_df.copy()

# Wetterdaten vorbereiten (Temperatur_Differenz_200 und Temperatur_Differenz_11, ohne NaNs)
wetter_df = pd.concat(jahres_daten.values())
wetter_df = wetter_df[["Temperatur_Differenz_200", "Temperatur_Differenz_11"]].dropna()
wetter_df.index = pd.to_datetime(wetter_df.index)

# Warengruppen in Dictionary: {1: DataFrame, ..., 6: DataFrame}
warengruppen = {i: merged_df[merged_df["Warengruppe"] == i].copy() for i in range(1, 7)}


# Funktion zum Joinen & Filtern (hier z.B. mit Temperatur_Differenz_200, bei Bedarf anpassen)
def merge_and_filter(war_df, wetter_df, diff_col="Temperatur_Differenz_200"):
    war_df = war_df.copy()

    # Datum vorbereiten
    war_df["Datum"] = pd.to_datetime(war_df["Datum"])
    war_df.set_index("Datum", inplace=True)

    # Nur gültige Umsatzdaten
    war_df = war_df[["Umsatz"]].dropna()

    # Inner Join per Datum
    kombi = war_df.join(wetter_df[[diff_col]], how="inner")

    # Nur relevante Spalten und Datum als Spalte
    kombi = kombi.reset_index()[["Datum", diff_col, "Umsatz"]]
    return kombi


# Beispiel: Kombinierte DataFrames für beide Fenstergrößen erstellen
warengruppe_merged_200 = {
    i: merge_and_filter(df, wetter_df, diff_col="Temperatur_Differenz_200")
    for i, df in warengruppen.items()
}

warengruppe_merged_11 = {
    i: merge_and_filter(df, wetter_df, diff_col="Temperatur_Differenz_11")
    for i, df in warengruppen.items()
}

# Scatterplots speichern (für Fenster 200)
for i, df in warengruppe_merged_200.items():
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Temperatur_Differenz_200"], df["Umsatz"], alpha=0.6)
    plt.title(f"Scatterplot Warengruppe {i}: Temperatur_Differenz_200 vs Umsatz")
    plt.xlabel("Temperatur_Differenz_200")
    plt.ylabel("Umsatz")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plots/scatter_warengruppe_{i}_tempdiff_200.png")
    plt.close()

# Scatterplots speichern (für Fenster 11)
for i, df in warengruppe_merged_11.items():
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Temperatur_Differenz_11"], df["Umsatz"], alpha=0.6)
    plt.title(f"Scatterplot Warengruppe {i}: Temperatur_Differenz_11 vs Umsatz")
    plt.xlabel("Temperatur_Differenz_11")
    plt.ylabel("Umsatz")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plots/scatter_warengruppe_{i}_tempdiff_11.png")
    plt.close()


# Korrelationsberechnung (Pearson) für Fenster 200
correlations_200 = {}

for i, df in warengruppe_merged_200.items():
    df_clean = df.dropna(subset=["Temperatur_Differenz_200", "Umsatz"])

    if len(df_clean) > 1:
        corr, p_value = pearsonr(
            df_clean["Temperatur_Differenz_200"], df_clean["Umsatz"]
        )
        correlations_200[i] = {"correlation": corr, "p_value": p_value}
    else:
        correlations_200[i] = {"correlation": None, "p_value": None}

# Korrelationsberechnung (Pearson) für Fenster 11
correlations_11 = {}

for i, df in warengruppe_merged_11.items():
    df_clean = df.dropna(subset=["Temperatur_Differenz_11", "Umsatz"])

    if len(df_clean) > 1:
        corr, p_value = pearsonr(
            df_clean["Temperatur_Differenz_11"], df_clean["Umsatz"]
        )
        correlations_11[i] = {"correlation": corr, "p_value": p_value}
    else:
        correlations_11[i] = {"correlation": None, "p_value": None}


# Ergebnisse anzeigen
print("Korrelationswerte für Fenstergröße 200:")
for i, stats in correlations_200.items():
    print(
        f"Warengruppe {i}: Pearson r = {stats['correlation']:.4f}, p-value = {stats['p_value']:.4g}"
    )

print("\nKorrelationswerte für Fenstergröße 11:")
for i, stats in correlations_11.items():
    print(
        f"Warengruppe {i}: Pearson r = {stats['correlation']:.4f}, p-value = {stats['p_value']:.4g}"
    )
