import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from pymongo import MongoClient
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import argparse

# Lesen der Cosmos-Verbindungszeichenfolge aus der Umgebungsvariable
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--uri', required=True, help="MongoDB URI with username/password")
parser.add_argument('-c', '--connection', required=True, help="Azure Blob Storage connection string")
args = parser.parse_args()


def load_data_from_cosmos(collection_name, uri):
    # Verbindung zur Cosmos DB herstellen
    client = MongoClient(uri)
    db = client['mdm']
    collection = db[collection_name]

    # Daten aus der angegebenen Collection abrufen
    cursor = collection.find()

    # Daten in ein DataFrame laden
    df = pd.DataFrame(list(cursor))

    return df

def train_and_save_model(data_df, model_file, container_prefix, storage_connection_string):
    # Datenbereinigung
    data_df['Bevölkerung'] = data_df['Bevölkerung'].str.replace(' ', '').astype(float)
    
    # Aufteilen der Features (Jahr) und der Zielvariablen (Bevölkerung)
    X = data_df[['Jahr']].values
    y = data_df['Bevölkerung'].values

    # Initialisieren und trainieren des Modells
    model = LinearRegression()
    model.fit(X, y)

    # Speichern des trainierten Modells
    joblib.dump(model, model_file)

    # Verbindung zum Azure Blob Storage herstellen
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

    # Neue Container-Namen erstellen und überprüfen, ob sie bereits existieren
    i = 1
    while True:
        container_name = f"{container_prefix}-{i}"
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            break
        i += 1

    # Container erstellen, wenn er nicht existiert
    container_client.create_container()

    # Blob-Client für das Modell erstellen
    blob_client = container_client.get_blob_client(model_file)

    # Modell in Blob Storage hochladen
    with open(model_file, "rb") as data:
        blob_client.upload_blob(data)

# Trainiere und lade das Modell für die Trainingsdaten hoch
training_data = load_data_from_cosmos("Trainingdata", args.uri)
train_and_save_model(training_data, r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models\trained_model_trainingdata.pkl', 'trainingdata-models', args.connection)

# Trainiere und lade das Modell für die Validierungsdaten hoch
validation_data = load_data_from_cosmos("Validationdata", args.uri)
train_and_save_model(validation_data, r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models\trained_model_validationdata.pkl', 'validationdata-models', args.connection)
