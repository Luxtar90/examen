# Reconocimiento de Vehículos en Video Usando Inteligencia Artificial

## Descripción General
Este proyecto utiliza técnicas de Machine Learning y Deep Learning para procesar videos y clasificar vehículos en las siguientes categorías:

- Autos
- Camionetas
- Camiones
- Motos

La solución emplea modelo entrenado en el momento, junto con algoritmos personalizados, para procesar videos y generar resultados detallados, incluyendo un video de salida con las clasificaciones superpuestas y un análisis de métricas del modelo.

---

## Características Clave

1. **Entrenamiento del Modelo:**
   - Dataset preparado a partir del conjunto de datos COCO.
   - Uso de modelos para detección de objetos.
   - Uso de scikit-learn para clasificación de vehículos.

2. **Procesamiento del Video:**
   - OpenCV para extraer cuadros y procesar el video.
   - YOLOv3-tiny configurado para detección rápida y precisa.
   - Superposición de etiquetas de clasificación en tiempo real sobre los vehículos detectados.

3. **Resultados y Métricas:**
   - Métricas como Precisión, Recall, F1-Score y matriz de confusión.
   - Video procesado mostrando las detecciones y clasificaciones.

4. **Generación de Reportes:**
   - Gráficos y resultados exportados a un archivo PDF.

---

## Estructura del Proyecto

```
examen/
├── data/
│   ├── annotations/         # Anotaciones del dataset
│   ├── video_input/         # Videos de entrada
│   ├── video_output/        # Videos procesados
│   ├── X_train.npy          # Características de entrenamiento
│   ├── y_train.npy          # Etiquetas de entrenamiento
├── models/
│   ├── random_forest_model.pkl  # Modelo de clasificación Random Forest
│   ├── yolo/                 # Configuraciones y pesos YOLO
│       ├── yolov3-tiny.cfg
│       ├── yolov3-tiny.weights
│       ├── coco.names
├── notebooks/               # Notebooks Jupyter para experimentos
├── src/
│   ├── video/
│   │   ├── video_processor.py   # Procesamiento del video principal
│   │   ├── yolo_detector.py     # Lógica de detección usando YOLO
│   ├── utils.py                # Funciones auxiliares
├── .gitignore
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
```

---

## Requisitos Previos

1. **Software Necesario:**
   - Python 3.8 o superior.
   - pip para instalación de paquetes.

2. **Dependencias:**
   Todas las dependencias se encuentran listadas en `requirements.txt`. Para instalarlas:

   ```bash
   pip install -r requirements.txt
   ```

3. **Pesos y Configuraciones YOLO:**
   Asegúrate de que los siguientes archivos estén disponibles en `models/yolo/`:
   - `yolov3-tiny.cfg`
   - `yolov3-tiny.weights`
   - `coco.names`

---

## Instrucciones de Ejecución

1. **Clonar el Repositorio:**
   ```bash
   git clone https://github.com/Luxtar90/examen.git
   cd examen
   ```

2. **Configurar el Entorno:**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   .\venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Ejecutar el Procesamiento de Video:**
   ```bash
   python src/video/video_processor.py
   ```

   - El video procesado se guardará en `data/video_output/`.

---

## YOLO: Configuración y Uso

- **Modelo YOLOv3-tiny:**
  Este modelo preentrenado se utiliza para detectar vehículos en cada cuadro del video.

- **Configuraciones Requeridas:**
  - `yolov3-tiny.cfg`: Configuración de la arquitectura.
  - `yolov3-tiny.weights`: Pesos del modelo preentrenado.
  - `coco.names`: Lista de clases que el modelo puede detectar.

- **Flujo de Procesamiento:**
  1. OpenCV lee los cuadros del video.
  2. YOLO detecta objetos en cada cuadro.
  3. Se filtran las detecciones para categorías de vehículos.
  4. Se aplican etiquetas sobre el cuadro y se genera un nuevo video.

---

## Resultados Esperados

- **Video Procesado:**
  Los vehículos detectados son destacados en el video con etiquetas indicando la clasificación.





---

## Contribuciones

Las contribuciones son bienvenidas. Por favor, realiza un fork del repositorio, haz tus cambios y envía un pull request.

---

## Licencia

Este proyecto está licenciado bajo la MIT License. Ver el archivo LICENSE para más información.


