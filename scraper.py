from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://de.m.wikipedia.org/wiki/Demografie_Deutschlands#Bev%C3%B6lkerungsr%C3%BCckgang_bis_2060'
page = requests.get(url)

soup = BeautifulSoup(page.text, 'html.parser')

tables = soup.find_all('table')

# Überprüfen, ob mindestens vier Tabellen auf der Seite vorhanden sind
if len(tables) >= 4:
    data_rows = []
    table = tables[3]  # Die vierte Tabelle hat den Index 3
    rows = table.find_all('tr')
    for row in rows:
        row_data = [cell.text.strip() for cell in row.find_all('td')][:2]  # Nur die ersten beiden Spalten auswählen
        if row_data:  # Ignoriere leere Zeilen
            data_rows.append(row_data)

    if data_rows:
        df = pd.DataFrame(data_rows, columns=['Jahr', 'Bevölkerung'])  # Benenne die Spalten entsprechend

        # Aufteilen der Daten in Trainings- und Validierungsdaten
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Speichern der Trainingsdaten
        train_save_path = r'C:/Users/thasm/Desktop/Model Deployment & Maintenance/Projekt1/Daten/Trainingdata.csv'
        train_df.to_csv(train_save_path, index=False)
        print(f"Die Trainingsdaten wurden erfolgreich in die Datei '{train_save_path}' gespeichert.")

        # Speichern der Validierungsdaten
        val_save_path = r'C:/Users/thasm/Desktop/Model Deployment & Maintenance/Projekt1/Daten/Validationdata.csv'
        val_df.to_csv(val_save_path, index=False)
        print(f"Die Validierungsdaten wurden erfolgreich in die Datei '{val_save_path}' gespeichert.")
    else:
        print("Keine Daten gefunden in der vierten Tabelle.")
else:
    print("Es sind nicht genügend Tabellen auf der Seite vorhanden.")
