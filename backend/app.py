from flask import Flask, send_file, request
import joblib
import os
from azure.storage.blob import BlobServiceClient
from connection2 import connectionStorage


# Laden des Flask-App-Objekts
app = Flask(__name__, static_url_path='', static_folder='../frontend')

# Funktion zum Laden des Modells aus dem Azure Blob Storage
def load_model_from_blob(container_name, model_file, save_path, new_model_name):
    azure_storage_connection_string = connectionStorage
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(model_file)
    os.makedirs(save_path, exist_ok=True)
    local_model_path = os.path.join(save_path, new_model_name)
    with open(local_model_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().read())
    return joblib.load(local_model_path)

# Laden der trainierten Modelle aus Azure Blob Storage
model_trainingdata = load_model_from_blob('trainingdata-models-1', 'trained_model_trainingdata.pkl', 
                                          r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models', 
                                          'AZmodel_trainingdata.pkl')
model_validationdata = load_model_from_blob('validationdata-models-1', 'trained_model_validationdata.pkl', 
                                            r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models', 
                                            'AZmodel_validationdata.pkl')

# Routen definieren
@app.route("/")
def indexPage():
    return send_file("../frontend/index.html")

@app.route("/predict", methods=["GET"])
def predict():
    # Benutzereingabe für das Jahr aus der GET-Anfrage erhalten
    year = request.args.get('year')
    if not year:
        return "Please provide 'year' parameter in the URL"
    
    # Vorhersage mit dem Modell für Trainingsdaten und Runden auf ganze Zahl
    prediction_trainingdata = round(model_trainingdata.predict([[int(year)]])[0])
    print(f"Vorhersage mit dem Modell für Trainingsdaten: {prediction_trainingdata}")
    
    # Vorhersage mit dem Modell für Validierungsdaten und Runden auf ganze Zahl
    prediction_validationdata = round(model_validationdata.predict([[int(year)]])[0])
    print(f"Vorhersage mit dem Modell für Validierungsdaten: {prediction_validationdata}")
    
    # Rückgabe der Vorhersagen als JSON
    return {
        "prediction_trainingdata": prediction_trainingdata,
        "prediction_validationdata": prediction_validationdata
    }
    
@app.route("/test_prediction")
def test_prediction():
    # Hier rufe die Vorhersagefunktionen auf
    year = 2030  # Ein Beispieljahr für den Test
    prediction_trainingdata = round(model_trainingdata.predict([[year]])[0])
    prediction_validationdata = round(model_validationdata.predict([[year]])[0])

    # Gib die Vorhersagen zurück
    return {
        "prediction_trainingdata": prediction_trainingdata,
        "prediction_validationdata": prediction_validationdata
    }
    

# Hauptfunktion zum Ausführen der Flask-App
if __name__ == "__main__":
    app.run(debug=True)
