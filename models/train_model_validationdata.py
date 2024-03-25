import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Lade Validierungsdaten
val_data = pd.read_csv('Validationdata.csv')

# Bereinige und konvertiere Bevölkerungszahlen in numerische Werte
val_data['Bevölkerung'] = val_data['Bevölkerung'].str.replace(' ', '').astype(float)

# Aufteilen der Features (Jahr) und der Zielvariablen (Bevölkerung)
X_val = val_data[['Jahr']].values
y_val = val_data['Bevölkerung'].values

# Initialisieren und trainieren des Modells
model = LinearRegression()
model.fit(X_val, y_val)

# Speichern des trainierten Modells
joblib.dump(model, 'trained_model_validationdata.pkl')
