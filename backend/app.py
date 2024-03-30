from flask import Flask, send_file, request, jsonify
import joblib
import os
from azure.storage.blob import BlobServiceClient

# Laden des Flask-App-Objekts
app = Flask(__name__, static_url_path='', static_folder='../frontend')

# Funktion zum Laden des Modells aus dem Azure Blob Storage
def load_model_from_blob(container_name, model_file, save_path, new_model_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(model_file)
    os.makedirs(save_path, exist_ok=True)
    local_model_path = os.path.join(save_path, new_model_name)
    with open(local_model_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().read())
    return joblib.load(local_model_path)

# Laden der trainierten Modelle aus Azure Blob Storage
azure_connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
if not azure_connection_string:
    raise ValueError("Azure Blob Storage connection string is not provided in the environment variable 'AZURE_STORAGE_CONNECTION_STRING'.")

model_Linear = load_model_from_blob('linear-model-data1', 'linear-model.pkl', 
                                          r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models', 
                                          'AZmodel_linearModel.pkl', azure_connection_string)
model_RandomForest = load_model_from_blob('random-forest-model-data1', 'random-forest-model.pkl', 
                                            r'C:\Users\thasm\Desktop\Model Deployment & Maintenance\Projekt1\models', 
                                            'AZmodel_RandomForest.pkl', azure_connection_string)

# Funktion zum Laden der MSE-Dateien aus dem Azure Blob Storage
def load_mse_from_blob(container_name, mse_files, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    mse_contents = {}
    for mse_file in mse_files:
        blob_client = container_client.get_blob_client(mse_file)
        mse_contents[mse_file] = blob_client.download_blob().content_as_text()
    return mse_contents

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
    prediction_Linearmodel = round(model_Linear.predict([[int(year)]])[0])
    print(f"Vorhersage mit dem Modell für Trainingsdaten: {prediction_Linearmodel}")
    
    # Vorhersage mit dem Modell für Validierungsdaten und Runden auf ganze Zahl
    prediction_RandomForest = round(model_RandomForest.predict([[int(year)]])[0])
    print(f"Vorhersage mit dem Modell für Validierungsdaten: {prediction_RandomForest}")
    
    # Rückgabe der Vorhersagen als JSON
    return {
        "prediction_LinearModel": prediction_Linearmodel,
        "prediction_RandomForest": prediction_RandomForest
    }
    
@app.route("/test_prediction")
def test_prediction():
    # Vorhersagefunktion aufrufen
    year = 2030  # Ein Beispieljahr für den Test
    prediction_Linearmodel = round(model_Linear.predict([[year]])[0])
    prediction_RandomForest = round(model_RandomForest.predict([[year]])[0])

    # Vorhersage zurückgeben
    return {
        "prediction_Linearmodel": prediction_Linearmodel,
        "prediction_RandomForest": prediction_RandomForest
    }

@app.route("/mse")
def get_mse():
    # Laden der MSE-Dateien für das lineare Modell
    linear_mse_files = ['train_mse.txt', 'valid_mse.txt']
    linear_mse_contents = load_mse_from_blob('linear-mse-data1', linear_mse_files, azure_connection_string)

    # Laden der MSE-Dateien für das Random-Forest-Modell
    random_forest_mse_files = ['train_mse_rf.txt', 'valid_mse_rf.txt']
    random_forest_mse_contents = load_mse_from_blob('random-forest-mse-data1', random_forest_mse_files, azure_connection_string)

    # Zusammenführen der MSE-Dateien und Rückgabe als JSON
    mse_data = {
        "linear_model": linear_mse_contents,
        "random_forest_model": random_forest_mse_contents
        
    }

    return jsonify(mse_data)

# Hauptfunktion zum Ausführen der Flask-App
if __name__ == "__main__":
    app.run(debug=True)
