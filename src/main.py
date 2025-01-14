from src.process.dataset_preprocessor import preprocess_dataset
from src.train.train_sklearn_model import train_random_forest
from src.video.video_processor import process_video

def main():
    print("Seleccione una opción:")
    print("1. Preprocesar datos")
    print("2. Entrenar modelo")
    print("3. Procesar video")
    option = input("Ingrese la opción (1/2/3): ")

    if option == "1":
        preprocess_dataset("../data/vehicles", "../data", (224, 224), 0.2)
    elif option == "2":
        train_random_forest("../data/X_train.npy", "../data/y_train.npy", "../models/random_forest_model.pkl")
    elif option == "3":
        process_video("../data/video_input/videoplayback.mp4",
                      "../data/video_output/output.mp4",
                      "../models/random_forest_model.pkl",
                      "../data/cars.xml")
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
