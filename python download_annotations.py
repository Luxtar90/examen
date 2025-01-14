import requests
import os

def download_file(url, output_path):
    """
    Descarga un archivo desde una URL y lo guarda en la ruta especificada.
    
    :param url: URL del archivo a descargar.
    :param output_path: Ruta donde se guardará el archivo descargado.
    """
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # Tamaño del bloque para mostrar progreso
    progress_bar = 0

    with open(output_path, 'wb') as file:
        print(f"Descargando {os.path.basename(output_path)}...")
        for data in response.iter_content(block_size):
            progress_bar += len(data)
            file.write(data)
            print(f"\rProgreso: {progress_bar / file_size:.2%}", end="")
    
    print("\nDescarga completa!")

# Configuración
url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
output_path = "./annotations_trainval2017.zip"

# Llamar a la función para descargar el archivo
download_file(url, output_path)

# Si necesitas descomprimirlo después:
import zipfile

with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("./data/annotations")
    print("Archivo descomprimido en ./data/annotations")
