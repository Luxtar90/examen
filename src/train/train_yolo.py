from ultralytics import YOLO

def train_yolo(data_yaml, model_name='yolov8n.pt', epochs=50, img_size=640, output_dir='runs/train'):
    """
    Entrena un modelo YOLO con un dataset personalizado.

    :param data_yaml: Ruta al archivo YAML que describe el dataset.
    :param model_name: Modelo base preentrenado de YOLO.
    :param epochs: Número de épocas para el entrenamiento.
    :param img_size: Tamaño de las imágenes de entrada.
    :param output_dir: Directorio para guardar los resultados del entrenamiento.
    """
    # Cargar el modelo YOLO preentrenado
    model = YOLO(model_name)

    # Entrenar el modelo con el dataset personalizado
    print(f"Iniciando el entrenamiento con {model_name}...")
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, project=output_dir)

    print("Entrenamiento completado. Los resultados se guardaron en:", output_dir)

if __name__ == "__main__":
    # Configuración
    DATA_YAML = "./data/yolo/data.yaml"  # Ruta al archivo YAML que define el dataset
    MODEL_NAME = "yolov8n.pt"           # Puedes usar 'yolov8n.pt' o 'yolov8s.pt'
    EPOCHS = 50                         # Número de épocas
    IMG_SIZE = 640                      # Tamaño de las imágenes de entrada

    # Entrenar el modelo YOLO
    train_yolo(DATA_YAML, MODEL_NAME, EPOCHS, IMG_SIZE)
