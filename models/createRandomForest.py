import argparse
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# Lesen der Cosmos- und Azure Blob Storage Verbindungszeichenfolgen aus der Umgebungsvariable
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--uri', required=True, help="Cosmos DB URI with username/password")
parser.add_argument('-c', '--connection', required=True, help="Azure Blob Storage connection string")
args = parser.parse_args()

def load_data_from_cosmos(collection_name, uri):
    # Verbindung zur Cosmos DB herstellen
    client = MongoClient(uri)
    db = client['mdm']  # Datenbank "mdm" auswählen

    # Daten aus der angegebenen Sammlung abrufen
    collection = db[collection_name]
    cursor = collection.find()

    # Daten in ein Pandas DataFrame laden und nur die erforderlichen Features auswählen
    data = pd.DataFrame(list(cursor), columns=["Jahr", "Bevölkerung"])

    # Bevölkerungszahlen von Zeichenfolgen in numerische Werte konvertieren
    data["Bevölkerung"] = data["Bevölkerung"].str.replace(" ", "").astype(int)

    return data

# Laden der Trainings- und Validierungsdaten aus Cosmos DB
train_data = load_data_from_cosmos("Trainingdata", args.uri)
valid_data = load_data_from_cosmos("Validationdata", args.uri)

# Aufteilen der Features für Trainings- und Validierungsdaten
X_train = train_data[["Jahr"]]
y_train = train_data["Bevölkerung"]

X_valid = valid_data[["Jahr"]]
y_valid = valid_data["Bevölkerung"]

# Initialisieren des Random Forest Regressionsmodells mit einigen Grundparametern
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Trainieren des Modells mit den Trainingsdaten
random_forest_model.fit(X_train, y_train)

# Vorhersagen auf den Trainings- und Validierungsdaten
train_predictions = random_forest_model.predict(X_train)
valid_predictions = random_forest_model.predict(X_valid)

# Auswertung der Leistung des Modells auf den Trainings- und Validierungsdaten (MSE)
train_mse = mean_squared_error(y_train, train_predictions)
valid_mse = mean_squared_error(y_valid, valid_predictions)

print("Trainings-MSE:", train_mse)
print("Validierungs-MSE:", valid_mse)

# Speichern des trainierten Modells als .pkl-Datei
model_file_name = "random-forest-model.pkl"
joblib.dump(random_forest_model, model_file_name)

# Speichern der MSE-Werte in Textdateien
with open('train_mse_rf.txt', 'w') as train_mse_file:
    train_mse_file.write(str(train_mse))

with open('valid_mse_rf.txt', 'w') as valid_mse_file:
    valid_mse_file.write(str(valid_mse))

# Verbindung zum Azure Blob Storage herstellen
blob_service_client = BlobServiceClient.from_connection_string(args.connection)

# Hochladen des trainierten Modells in Azure Blob Storage
container_counter_model = 1
container_name_model = f"random-forest-model-data{container_counter_model}"
while True:
    try:
        blob_service_client.create_container(container_name_model)
        break
    except ResourceExistsError:
        container_counter_model += 1
        container_name_model = f"random-forest-model-data{container_counter_model}"

blob_client_model = blob_service_client.get_blob_client(container=container_name_model, blob=model_file_name)
with open(model_file_name, "rb") as data:
    blob_client_model.upload_blob(data)

print(f"Das Modell wurde erfolgreich als {model_file_name} in den Container {container_name_model} hochgeladen.")

# Hochladen der MSE-Werte in Azure Blob Storage
container_counter_mse = 1
container_name_mse = f"random-forest-mse-data{container_counter_mse}"
while True:
    try:
        blob_service_client.create_container(container_name_mse)
        break
    except ResourceExistsError:
        container_counter_mse += 1
        container_name_mse = f"random-forest-mse-data{container_counter_mse}"

blob_client_train_mse = blob_service_client.get_blob_client(container=container_name_mse, blob='train_mse_rf.txt')
with open('train_mse_rf.txt', "rb") as data:
    blob_client_train_mse.upload_blob(data)

blob_client_valid_mse = blob_service_client.get_blob_client(container=container_name_mse, blob='valid_mse_rf.txt')
with open('valid_mse_rf.txt', "rb") as data:
    blob_client_valid_mse.upload_blob(data)

print(f"Die MSE-Werte wurden erfolgreich in den Container {container_name_mse} hochgeladen.")
