import joblib

# Laden der trainierten Modelle
model_trainingdata = joblib.load('trained_model_trainingdata.pkl')
model_validationdata = joblib.load('trained_model_validationdata.pkl')

# Benutzereingabe für das Jahr, das vorhergesagt werden soll
year = int(input("Geben Sie das Jahr ein, für das Sie die Bevölkerung vorhersagen möchten: "))

# Vorhersage mit dem Modell für Trainingsdaten und Runden auf ganze Zahl
prediction_trainingdata = round(model_trainingdata.predict([[year]])[0])
print(f"Vorhersage mit dem Modell für Trainingsdaten: {prediction_trainingdata}")

# Vorhersage mit dem Modell für Validierungsdaten und Runden auf ganze Zahl
prediction_validationdata = round(model_validationdata.predict([[year]])[0])
print(f"Vorhersage mit dem Modell für Validierungsdaten: {prediction_validationdata}")
