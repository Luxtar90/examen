import os
import json
import shutil

def organize_images_by_category(annotations_file, input_folder, output_folder, categories):
    """
    Organiza las imágenes en subcarpetas por categoría, según las anotaciones.
    
    :param annotations_file: Ruta al archivo de anotaciones (JSON).
    :param input_folder: Carpeta donde están todas las imágenes descargadas.
    :param output_folder: Carpeta de salida para organizar las imágenes por categoría.
    :param categories: Lista de categorías de interés (por ejemplo: ['car', 'motorcycle', 'truck']).
    """
    # Cargar las anotaciones
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Crear un diccionario para mapear categorías
    category_mapping = {category['id']: category['name'] for category in annotations['categories']}
    category_ids = {cat_id: name for cat_id, name in category_mapping.items() if name in categories}

    # Crear carpetas para cada categoría
    os.makedirs(output_folder, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(output_folder, category), exist_ok=True)

    # Reorganizar las imágenes basadas en las categorías
    for annotation in annotations['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']

        if category_id in category_ids:
            category_name = category_ids[category_id]
            for image in annotations['images']:
                if image['id'] == image_id:
                    image_name = image['file_name']
                    source_path = os.path.join(input_folder, image_name)
                    dest_path = os.path.join(output_folder, category_name, image_name)

                    if os.path.exists(source_path):
                        shutil.move(source_path, dest_path)
                        print(f"Moviendo {image_name} a {category_name}/")

    print("Organización completada.")

# Configuración
ANNOTATIONS_FILE = './data/annotations/instances_train2017.json'
INPUT_FOLDER = './data/vehicles'
OUTPUT_FOLDER = './data/organized_vehicles'
CATEGORIES = ['car', 'motorcycle', 'truck']

if __name__ == "__main__":
    organize_images_by_category(ANNOTATIONS_FILE, INPUT_FOLDER, OUTPUT_FOLDER, CATEGORIES)
