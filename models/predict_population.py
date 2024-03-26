import joblib
import os
from azure.storage.blob import BlobServiceClient
from connection import connectionStorage



def load_model_from_blob(container_name, model_file, save_path, new_model_name):
    # Verbindung zum Azure Blob Storage herstellen
    azure_storage_connection_string = connectionStorage
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)

    # Container Client erstellen oder referenzieren
    container_client = blob_service_client.get_container_client(container_name)

    # Blob-Client für das Modell erstellen
    blob_client = container_client.get_blob_client(model_file)

    # Pfad für das Speichern des heruntergeladenen Modells erstellen
    os.makedirs(save_path, exist_ok=True)
    local_model_path = os.path.join(save_path, new_model_name)

    # Modell aus Blob Storage herunterladen und speichern
    with open(local_model_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().read())

    # Modell laden und zurückgeben
    return joblib.load(local_model_path)

# Laden der trainierten Modelle aus den Azure Blob Containern und Speichern im gewünschten Ordner
model_trainingdata = load_model_from_blob('trainingdata-models-1', 'trained_model_trainingdata.pkl', 
                                          r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models', 
                                          'AZmodel_trainingdata.pkl')
model_validationdata = load_model_from_blob('validationdata-models-1', 'trained_model_validationdata.pkl', 
                                            r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models', 
                                            'AZmodel_validationdata.pkl')

# Benutzereingabe für das Jahr, das vorhergesagt werden soll
year = int(input("Geben Sie das Jahr ein, für das Sie die Bevölkerung vorhersagen möchten: "))

# Vorhersage mit dem Modell für Trainingsdaten und Runden auf ganze Zahl
prediction_trainingdata = round(model_trainingdata.predict([[year]])[0])
print(f"Vorhersage mit dem Modell für Trainingsdaten: {prediction_trainingdata}")

# Vorhersage mit dem Modell für Validierungsdaten und Runden auf ganze Zahl
prediction_validationdata = round(model_validationdata.predict([[year]])[0])
print(f"Vorhersage mit dem Modell für Validierungsdaten: {prediction_validationdata}")
