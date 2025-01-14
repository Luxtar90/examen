import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(X_train_path, y_train_path, model_output_path):
    # Verificar que los archivos existen
    if not os.path.exists(X_train_path):
        raise FileNotFoundError(f"No se encontró el archivo: {X_train_path}")
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"No se encontró el archivo: {y_train_path}")

    # Cargar los datos
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    # Ajustar la forma de los datos si es necesario
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)

    print(f"Datos cargados: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")

    # Crear y entrenar el modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Modelo entrenado con éxito.")

    # Guardar el modelo entrenado
    joblib.dump(clf, model_output_path)
    print(f"Modelo guardado en: {model_output_path}")

if __name__ == "__main__":
    train_random_forest("./data/X_train.npy", "./data/y_train.npy", "./models/random_forest_model.pkl")
