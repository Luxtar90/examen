import os
import joblib

model_path = "./models/random_forest_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

# Cargar el modelo
model = joblib.load(model_path)
print("Modelo cargado con éxito.")
