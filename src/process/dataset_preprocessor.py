import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_dataset(input_path, output_path, img_size=(224, 224), test_size=0.2):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el directorio: {input_path}")
    
    categories = os.listdir(input_path)
    data = []
    labels = []

    for idx, category in enumerate(categories):
        category_path = os.path.join(input_path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            img = cv2.imread(file_path)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                data.append(img_resized)
                labels.append(idx)

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

    np.save(os.path.join(output_path, "X_train.npy"), X_train)
    np.save(os.path.join(output_path, "X_test.npy"), X_test)
    np.save(os.path.join(output_path, "y_train.npy"), y_train)
    np.save(os.path.join(output_path, "y_test.npy"), y_test)

    print(f"Datos preprocesados y guardados en: {output_path}")

if __name__ == "__main__":
    preprocess_dataset("./data/vehicles", "./data", (224, 224), 0.2)
