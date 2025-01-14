import cv2
import numpy as np
import os

def process_video_with_yolo(video_path, output_path, yolo_weights, yolo_cfg, yolo_names):
    print(f"Ruta del video: {video_path}")
    print(f"Ruta del YOLO weights: {yolo_weights}")
    print(f"Ruta del YOLO cfg: {yolo_cfg}")
    print(f"Ruta del YOLO names: {yolo_names}")

    # Verificar existencia de archivos
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")
    if not os.path.exists(yolo_weights) or not os.path.exists(yolo_cfg) or not os.path.exists(yolo_names):
        raise FileNotFoundError("Faltan archivos del modelo random_forest_model.pkl (weights, cfg o names)")

    print("Cargando modelo random_forest_model.pkl ...")
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    with open(yolo_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("Modelo random_forest_model.pkl cargado con éxito.")

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    print("Procesando video...")
    frame_count = 0  # Para depuración
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video.")
            break

        frame_count += 1
        print(f"Procesando cuadro {frame_count}...")

        # Preprocesar el cuadro para YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        detections = net.forward(output_layers)

        # Detección con YOLO
        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            if isinstance(output, np.ndarray):  # Verificar que la salida sea una matriz válida
                for detection in output:
                    scores = detection[5:]
                    class_id = int(np.argmax(scores))
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Coordenadas de la detección
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        w = int(detection[2] * frame_width)
                        h = int(detection[3] * frame_height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if indexes is not None and len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Procesando Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Procesamiento interrumpido por el usuario.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video procesado guardado en: {output_path}")

# Configuración
video_input = r"C:\Users\luigg\OneDrive\Desktop\examen\data\video_input\videoplayback.mp4"
video_output = r"C:\Users\luigg\OneDrive\Desktop\examen\data\video_output\video_output.mp4"
yolo_weights = r"C:\Users\luigg\OneDrive\Desktop\examen\data\yolo\yolov3-tiny.weights"
yolo_cfg = r"C:\Users\luigg\OneDrive\Desktop\examen\data\yolo\yolov3-tiny.cfg"
yolo_names = r"C:\Users\luigg\OneDrive\Desktop\examen\data\yolo\coco.names"

if __name__ == "__main__":
    try:
        process_video_with_yolo(video_input, video_output, yolo_weights, yolo_cfg, yolo_names)
    except Exception as e:
        print(f"Error: {e}")
