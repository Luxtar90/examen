import requests

url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
output_path = "c:/Users/luigg/OneDrive/Desktop/examen/data/yolo/yolov3-tiny.weights"

print("Descargando yolov3-tiny.weights...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Descarga completada:", output_path)
else:
    print("Error al descargar el archivo:", response.status_code)
