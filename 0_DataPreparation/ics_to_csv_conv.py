from ics import Calendar
from datetime import datetime, timedelta
import pandas as pd

file_path = "External/sh_holidays.ics"

# ICS-Datei einlesen
with open(file_path, "r", encoding="utf-8") as f:
    kalender = Calendar(f.read())

# Dictionaries für die Markierungstage
feiertage = {}
ferientage = {}

# Alle Events durchgehen
for event in kalender.events:
    start_date = event.begin.date()
    end_date = event.end.date() - timedelta(days=1)  # ICS-Ende ist exklusiv

    # Über die Event-Tage iterieren
    for single_date in (
        start_date + timedelta(n) for n in range((end_date - start_date).days + 1)
    ):
        if "ferien" in event.name.lower():
            ferientage[single_date] = 1
        elif "feiertag" in event.name.lower():
            feiertage[single_date] = 1

# Alle betroffenen Tage sammeln
alle_daten = set(feiertage.keys()) | set(ferientage.keys())

# DataFrame erzeugen
df = pd.DataFrame({"Datum": list(alle_daten)})

df["Feiertag"] = df["Datum"].map(lambda d: feiertage.get(d, 0))
df["Ferientag"] = df["Datum"].map(lambda d: ferientage.get(d, 0))

# Nach Datum sortieren
df = df.sort_values("Datum")

# Datum als String formatieren
df["Datum"] = df["Datum"].astype(str)

# Als CSV speichern
df.to_csv("External/kalender_markiert.csv", index=False)
