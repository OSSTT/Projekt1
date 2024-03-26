from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

# Laden der Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Verbindung zur Cosmos DB herstellen
connection_string = os.getenv("CosmosConnectionString")
client = MongoClient(connection_string)

# Datenbank und Collections auswählen
db = client['mdm']
training_collection = db['Trainingdata']
validation_collection = db['Validationdata']

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

        # Daten direkt in Cosmos DB hochladen
        training_collection.insert_many(train_df.to_dict('records'))
        validation_collection.insert_many(val_df.to_dict('records'))

        print("Die Daten wurden erfolgreich in Azure Cosmos DB hochgeladen.")
    else:
        print("Keine Daten gefunden in der vierten Tabelle.")
else:
    print("Es sind nicht genügend Tabellen auf der Seite vorhanden.")
