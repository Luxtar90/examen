# Reconocimiento de Vehículos en Video Usando Inteligencia Artificial

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un sistema basado en inteligencia artificial que procese videos para reconocer y clasificar vehículos en las categorías: autos, camionetas, camiones y motos. Utilizando técnicas de visión por computadora y aprendizaje profundo, se analizan cuadros de video para detectar y clasificar los vehículos presentes.

---

## Requisitos del Sistema

Para ejecutar este proyecto, asegúrate de que tu sistema cumple con los siguientes requisitos:

### Lenguaje de Programación
- **Python** 3.8 o superior.

### Librerías Necesarias
Las siguientes librerías deben estar instaladas:
- **OpenCV** (procesamiento de video)
- **TensorFlow** o **PyTorch** (entrenamiento y ejecución del modelo)
- **scikit-learn** (evaluación y métricas del modelo)
- **NumPy** (operaciones matemáticas)
- **Matplotlib** (visualización de datos)
- **Pandas** (manejo de datos estructurados)

Instala las dependencias ejecutando:
```bash
pip install -r requirements.txt

examen/
├── data/
│   ├── annotations/          # Archivos de anotación (JSON)
│   ├── video_input/          # Videos de entrada para procesar
│   ├── video_output/         # Videos procesados de salida
│   ├── X_train.npy           # Dataset de entrenamiento
│   ├── y_train.npy           # Etiquetas de entrenamiento
│   ├── X_test.npy            # Dataset de prueba
│   └── y_test.npy            # Etiquetas de prueba
├── models/                   # Modelos entrenados guardados
├── notebooks/                # Jupyter Notebooks para análisis
├── src/
│   ├── preprocessing.py      # Código para preprocesamiento de datos
│   ├── train_model.py        # Código para entrenar el modelo
│   ├── video_processor.py    # Código para procesar el video
│   └── utils.py              # Funciones auxiliares
├── venv/                     # Entorno virtual (excluido del repo)
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Instrucciones del proyecto


