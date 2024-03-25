import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from pymongo import MongoClient

def get_data_from_cosmosdb(uri, database_name, collection_name):
    client = MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]
    
    # Daten aus der Cosmos DB abrufen und in ein DataFrame konvertieren
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))
    
    return data

# Funktion zum Laden der Daten und Trainieren des Modells
def train_and_save_model(data, model_file):
    # Bereinigen und Konvertieren der Bevölkerungszahlen in numerische Werte
    data['Bevölkerung'] = data['Bevölkerung'].str.replace(' ', '').astype(float)
    
    # Aufteilen der Features (Jahr) und der Zielvariablen (Bevölkerung)
    X = data[['Jahr']].values
    y = data['Bevölkerung'].values
    
    # Initialisieren und Trainieren des Modells
    model = LinearRegression()
    model.fit(X, y)
    
    # Speichern des trainierten Modells
    joblib.dump(model, model_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train and save model')
    parser.add_argument('-u', '--uri', required=True, help="MongoDB URI with username/password")
    args = parser.parse_args()

    # Verbindungsinformationen für die Cosmos DB
    mongo_uri = args.uri
    database_name = "mdm"
    
    # Trainingsdaten aus Cosmos DB abrufen und Modell trainieren und speichern
    training_data = get_data_from_cosmosdb(mongo_uri, database_name, "Trainingdata")
    train_and_save_model(training_data, 'trained_model_trainingdata.pkl')
    
    # Validierungsdaten aus Cosmos DB abrufen und Modell trainieren und speichern
    validation_data = get_data_from_cosmosdb(mongo_uri, database_name, "Validationdata")
    train_and_save_model(validation_data, 'trained_model_validationdata.pkl')
