import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from djitellopy import Tello

# Configuración
TARGET_CLASS = "couch"
FRAME_SAVE_PATH = "1.inicio/frames/"
MASK_SAVE_PATH = "1.inicio/masks/"
DETECTION_SAVE_PATH = "1.inicio/detections/"
METRICS_SAVE_PATH = "1.inicio/metrics.csv"
NO_DETECTION_TIMEOUT = 10  # segundos sin detectar el sofá para abortar

# Crear carpetas si no existen
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
os.makedirs(MASK_SAVE_PATH, exist_ok=True)
os.makedirs(DETECTION_SAVE_PATH, exist_ok=True)

# Inicializar modelo de segmentación
model = YOLO("yolov8n-seg.pt")

# Inicializar dron
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
print("[INFO] Dron despegado")

# Obtener primer frame para dimensiones
frame_read = tello.get_frame_read()
frame = frame_read.frame
frame_height, frame_width = frame.shape[:2]
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

# Inicializar métricas
metrics = []
frame_id = 0
start_time = time.time()
last_detection_time = start_time

tello.send_rc_control(0, 0, 0, 0)

try:
    while True:
        frame = frame_read.frame #Frame original capturado por la cámara del dron
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #frame convertido de BGR a RGB
        timestamp = time.time()
        frame_id += 1
        tello.send_rc_control(0, 0, 0, 0)

        results = model(frame_rgb)[0]

        sofa_detected = False
        mask_output = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)

        center_x = center_y = offset_x = offset_y = None
        centrado = False

        if results.masks is not None: #Máscaras devueltas por YOLOv8n-seg para cada objeto detectado
            for seg, cls in zip(results.masks.data, results.boxes.cls):
                cls_name = model.names[int(cls)]

                if cls_name == TARGET_CLASS:
                    sofa_detected = True
                    last_detection_time = timestamp

                    #Máscara binaria de un solo objeto
                    mask = seg.cpu().numpy().astype(np.uint8) * 255
                    frame_height, frame_width = frame.shape[:2]
                    mask = cv2.resize(mask, (frame_width, frame_height))  # <-- OJO: ancho, luego alto
                    mask = cv2.resize(mask, (frame_width, frame_height))
                    print(f"[DEBUG] frame: {frame.shape}, mask: {mask.shape}, mask_output: {mask_output.shape}")

                    # Máscara binaria del frame, combina las mask
                    mask_output = np.maximum(mask_output, mask)

                    # Calcular centroide
                    M = cv2.moments(mask)
                    if M["m00"] > 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        offset_x = center_x - frame_center_x
                        offset_y = center_y - frame_center_y

                        # Verificar centrado (tolerancia de 30 píxeles)
                        centrado = abs(offset_x) < 30

                    break  # Solo consideramos la primera detección válida

        # Guardar métricas
        metrics.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "sofa_detected": int(sofa_detected),
            "centroid_x": center_x if center_x is not None else -1,
            "centroid_y": center_y if center_y is not None else -1,
            "frame_center_x": frame_center_x,
            "frame_center_y": frame_center_y,
            "offset_x": offset_x if offset_x is not None else -1,
            "offset_y": offset_y if offset_y is not None else -1,
            "centrado": int(centrado)
        })

        # Guardar imágenes
        frame_path = f"{FRAME_SAVE_PATH}frame_{frame_id:04d}.jpg"
        mask_path = f"{MASK_SAVE_PATH}mask_{frame_id:04d}.png"
        cv2.imwrite(frame_path, frame_rgb)
        cv2.imwrite(mask_path, mask_output)

        # Guardar detección si hubo
        if sofa_detected:
            # Creamos una versión en color de la máscara (rojo)
            colored_mask = cv2.merge([mask_output * 0, mask_output * 0, mask_output])
            # Superponemos la máscara roja sobre el frame original
            overlay = cv2.addWeighted(frame_rgb, 1.0, colored_mask, 0.5, 0)
            # Guardamos la imagen con detección sobrepuesta
            cv2.imwrite(f"{DETECTION_SAVE_PATH}overlay_{frame_id:04d}.jpg", overlay)
            print(f"[INFO] '{TARGET_CLASS}' detectado en frame {frame_id}. Pasando a la segunda etapa...")
            break  # 🚨 Rompe el bucle al detectar el primer sofá

        # Condición de parada por tiempo sin detección
        if not sofa_detected and (timestamp - last_detection_time > NO_DETECTION_TIMEOUT):
            print(f"[INFO] No se detectó '{TARGET_CLASS}' durante {NO_DETECTION_TIMEOUT} segundos. Aterrizando.")
            tello.land()
            break

        time.sleep(0.1)

except KeyboardInterrupt:
    print("[INFO] Interrupción manual. Aterrizando...")
    tello.land()

finally:
    # Guardar métricas
    df = pd.DataFrame(metrics)
    df.to_csv(METRICS_SAVE_PATH, index=False)

    # Finalizar vuelo y cerrar conexión
    tello.streamoff()
    tello.end()
    print("[INFO] Vuelo finalizado. Datos guardados.")
