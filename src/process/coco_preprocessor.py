from pycocotools.coco import COCO
import os
from shutil import copyfile

def filter_coco_dataset(annotation_path, images_dir, output_dir, categories):
    """
    Filtra imágenes y anotaciones del dataset COCO según categorías específicas.
    
    :param annotation_path: Ruta al archivo JSON de anotaciones.
    :param images_dir: Ruta al directorio con imágenes.
    :param output_dir: Directorio donde se guardarán las imágenes filtradas.
    :param categories: Lista de categorías para filtrar.
    """
    # Cargar anotaciones COCO
    coco = COCO(annotation_path)

    # Obtener IDs de las categorías y de las imágenes relacionadas
    category_ids = coco.getCatIds(catNms=categories)
    image_ids = coco.getImgIds(catIds=category_ids)
    images = coco.loadImgs(image_ids)

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Copiar imágenes filtradas
    for img in images:
        img_path = os.path.join(images_dir, img["file_name"])
        if os.path.exists(img_path):
            copyfile(img_path, os.path.join(output_dir, img["file_name"]))
        else:
            print(f"Imagen no encontrada: {img_path}")

    print(f"Filtrado completado: {len(images)} imágenes guardadas en {output_dir}")

# Configuración
if __name__ == "__main__":
    ANNOTATION_PATH = "./data/annotations/instances_train2017.json"
    IMAGES_DIR = "./data/train2017"  # Cambia esto si tus imágenes están en otro directorio
    OUTPUT_DIR = "./data/filtered_images"
    CATEGORIES = ["car", "motorcycle", "truck", "bicycle", "person"]  # Categorías específicas

    filter_coco_dataset(ANNOTATION_PATH, IMAGES_DIR, OUTPUT_DIR, CATEGORIES)
