import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from djitellopy import Tello

# ======================
# CONFIGURACIÓN
# ======================
TARGET_CLASS = "couch"
FRAME_SAVE_PATH = "2.movement/frames/"
MASK_SAVE_PATH = "2.movement/masks/"
DETECTION_SAVE_PATH = "2.movement/detections/"
METRICS_SAVE_PATH = "2.movement/metrics.csv"

NO_DETECTION_TIMEOUT = 10    # segundos sin detectar el sofá para abortar
VELOCIDAD_LATERAL = 13       # entre -100 y 100
YAW_SUAVE = 27               # velocidad rotación suave
HUMBRAL_OFFSET = 25          # humbral de distancia entre centro y centroide aceptable
SEGUNDA_ETAPA_DURACION = 35  # segundos


# Crear carpetas si no existen
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
os.makedirs(MASK_SAVE_PATH, exist_ok=True)
os.makedirs(DETECTION_SAVE_PATH, exist_ok=True)

# ======================
# INICIALIZACIÓN
# ======================
model = YOLO("yolov8n-seg.pt")

tello = Tello()
tello.connect()
tello.streamon()
print(f"[INFO] Batería inicial: {tello.get_battery()}%")

tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)
print("[INFO] Dron despegado")

frame_read = tello.get_frame_read()
frame = frame_read.frame
frame_height, frame_width = frame.shape[:2]
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

metrics = []
frame_id = 0
start_time = time.time()
last_detection_time = start_time

def guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected, center_x, center_y, offset_x, offset_y, frame_center_x, frame_center_y):
    # Guardar métricas
    metrics.append({
        "frame_id": frame_id,
        "timestamp": time.time(),
        "sofa_detected": int(sofa_detected),
        "centroid_x": center_x if sofa_detected else -1,
        "centroid_y": center_y if sofa_detected else -1,
        "frame_center_x": frame_center_x,
        "frame_center_y": frame_center_y,
        "offset_x": offset_x if sofa_detected else -1,
        "offset_y": offset_y if sofa_detected else -1,
        "centrado": int(abs(offset_x) < HUMBRAL_OFFSET if sofa_detected else 0)
    })

    # Guardar imágenes
    cv2.imwrite(f"{FRAME_SAVE_PATH}frame_{frame_id:04d}.jpg", frame_rgb)
    cv2.imwrite(f"{MASK_SAVE_PATH}mask_{frame_id:04d}.png", mask_output)

    if sofa_detected:
        # Asegurar que mask_output tenga 3 canales
        mask_color = cv2.merge([np.zeros_like(mask_output), np.zeros_like(mask_output), mask_output])

        # Redimensionar por si acaso
        mask_color = cv2.resize(mask_color, (frame_rgb.shape[1], frame_rgb.shape[0]))

        # Asegurar tipo uint8
        mask_color = mask_color.astype(np.uint8)

        # También asegurarse de que frame_rgb es uint8
        if frame_rgb.dtype != np.uint8:
            frame_rgb = (frame_rgb * 255).astype(np.uint8)

        overlay = cv2.addWeighted(frame_rgb, 1.0, mask_color, 0.5, 0)
        cv2.imwrite(f"{DETECTION_SAVE_PATH}overlay_{frame_id:04d}.jpg", overlay)

try:
    # ======================
    # PRIMERA ETAPA: detección inicial
    # ======================
    while True:
        frame = frame_read.frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = time.time()
        frame_id += 1
        tello.send_rc_control(0, 0, 0, 0)

        results = model(frame_rgb)[0]
        sofa_detected = False
        mask_output = np.zeros((frame_height, frame_width), dtype=np.uint8)

        center_x = center_y = offset_x = offset_y = None
        centrado = False

        if results.masks is not None:
            for seg, cls in zip(results.masks.data, results.boxes.cls):
                cls_name = model.names[int(cls)]
                if cls_name == TARGET_CLASS:
                    sofa_detected = True
                    last_detection_time = timestamp
                    mask = seg.cpu().numpy().astype(np.uint8) * 255
                    mask = cv2.resize(mask, (frame_width, frame_height))
                    mask_output = np.maximum(mask_output, mask)

                    M = cv2.moments(mask)
                    if M["m00"] > 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        offset_x = center_x - frame_center_x
                        offset_y = center_y - frame_center_y
                        centrado = abs(offset_x) < HUMBRAL_OFFSET
                    break

        # Guardar métricas
        guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected, center_x, center_y, offset_x, offset_y, frame_center_x, frame_center_y)
        if sofa_detected:
            print(f"[INFO] '{TARGET_CLASS}' detectado en frame {frame_id}. ✅ PASANDO A LA SEGUNDA ETAPA ... ✅")
            break

        # Verificar si ha pasado demasiado tiempo sin detección
        if time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
            print("[WARN] No se detectó el sofá dentro del tiempo límite. Abortando.")
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            raise SystemExit

    # ======================
    # SEGUNDA ETAPA: movimiento lateral y corrección
    # ======================
    etapa2_start = time.time()
    while time.time() - etapa2_start < SEGUNDA_ETAPA_DURACION:
        frame = frame_read.frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = time.time()
        frame_id += 1

        results = model(frame_rgb)[0]
        sofa_detected = False
        offset_x = 0
        yaw_velocity = 0

        mask_output = np.zeros((frame_height, frame_width), dtype=np.uint8)
        center_x = center_y = offset_y = -1

        if results.masks is not None:
            for seg, cls in zip(results.masks.data, results.boxes.cls):
                cls_name = model.names[int(cls)]
                if cls_name == TARGET_CLASS:
                    sofa_detected = True
                    mask = seg.cpu().numpy().astype(np.uint8) * 255
                    mask = cv2.resize(mask, (frame_width, frame_height))
                    mask_output = np.maximum(mask_output, mask)

                    M = cv2.moments(mask)
                    if M["m00"] > 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        offset_x = center_x - frame_center_x
                        print('❗ offset de:', offset_x)
                        if abs(offset_x) > HUMBRAL_OFFSET:
                            yaw_velocity = int(np.clip(offset_x / 1.5, -YAW_SUAVE, YAW_SUAVE))
                    break

        tello.send_rc_control(-VELOCIDAD_LATERAL, 0, 0, yaw_velocity)

        # Guardar métricas
        guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected, center_x, center_y, offset_x, offset_y, frame_center_x, frame_center_y)
        time.sleep(0.1)

    print("[INFO] Segunda etapa completada. Aterrizando...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()

except KeyboardInterrupt:
    print("[INFO] Interrupción manual. Aterrizando...")
    tello.land()

finally:
    # Guardar métricas
    df = pd.DataFrame(metrics)
    df.to_csv(METRICS_SAVE_PATH, index=False)

    tello.streamoff()
    tello.end()
    print("[INFO] Vuelo finalizado. Datos guardados.")
