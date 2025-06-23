import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from djitellopy import Tello
from PIL import Image
import torchvision.transforms as T

# ======================
# CONFIGURACIÓN
# ======================
TARGET_CLASS = "couch"
FRAME_SAVE_PATH = "4.Detections/frames/"
MASK_SAVE_PATH = "4.Detections/masks/"
DETECTION_SAVE_PATH = "4.Detections/detections/"
METRICS_SAVE_PATH = "4.Detections/metrics.csv"
DEPTH_MAPS_SAVE_PATH = "4.Detections/depth_maps/"
MODELS_DETECTION_PATH = "4.Detections/models_detections/"

# NUEVO: ruta y umbral para modelos de partes de coche y desperfectos
MODEL_PARTS_PATH = r"C:\Users\USUARIO\TFG\modelos_pruebas\parts_best.pt"
MODEL_DEFECTS_PATH = r"C:\Users\USUARIO\TFG\modelos_pruebas\damages_best.pt"
CONF_THRESHOLD = 0.5  # confianza mínima para procesar detecciones

NO_DETECTION_TIMEOUT = 10    # segundos sin detectar el sofá para abortar
VELOCIDAD_LATERAL = 13       # entre -100 y 100
YAW_SUAVE = 25               # velocidad rotación suave
HUMBRAL_OFFSET = 60          # umbral de distancia entre centro y centroide aceptable
SEGUNDA_ETAPA_DURACION = 35  # segundos

PROFUNDIDAD_MIN = 2.3
PROFUNDIDAD_MAX = 2.7
PROFUNDIDAD_OBJETIVO = (PROFUNDIDAD_MIN + PROFUNDIDAD_MAX) / 2  # 2.5

# Crear carpetas si no existen
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
os.makedirs(MASK_SAVE_PATH, exist_ok=True)
os.makedirs(DETECTION_SAVE_PATH, exist_ok=True)
os.makedirs(DEPTH_MAPS_SAVE_PATH, exist_ok=True)
os.makedirs(MODELS_DETECTION_PATH, exist_ok=True)


# ======================
# INICIALIZACIÓN
# ======================
# Cargar modelo YOLO-seg para primera fase (segmentación de sofá)
model = YOLO("yolov8n-seg.pt")

# Dispositivo para MiDaS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo MiDaS
midas = torch.hub.load("isl-org/MiDaS", "DPT_LeViT_224", trust_repo=True)
midas.to(device)
midas.eval()

# Transformaciones para MiDaS
midas_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])
print("✅ MiDaS cargado correctamente.")

# ======================
# CARGAR MODELOS DE DETECCIÓN DE PARTES Y DES PERFECTOS
# ======================
model_parts = YOLO(MODEL_PARTS_PATH)
model_defects = YOLO(MODEL_DEFECTS_PATH)
print("✅ Modelos de partes y desperfectos cargados correctamente.")

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

# Almacenaremos dos listas de métricas:
# 1) Para control de profundidad/posición (como antes)
metrics = []
# 2) Para detecciones (partes y desperfectos)
detection_metrics = []

frame_id = 0
start_time = time.time()
last_detection_time = start_time

def guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected,
                       center_x, center_y, offset_x, offset_y,
                       frame_center_x, frame_center_y,
                       depth_value=None, depth_error=None):
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
        "centrado": int(abs(offset_x) < HUMBRAL_OFFSET if sofa_detected else 0),
        "depth_value": depth_value if depth_value is not None else -1,
        "depth_error": depth_error if depth_error is not None else -1
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

def estimate_masked_depth(frame, mask):
    # Asegurar que mask sea binaria y del mismo tamaño que frame
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Enmascarar la imagen antes de pasarla a MiDaS
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convertir a RGB y preparar para PIL
    img_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Transformar con MiDaS
    input_tensor = midas_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Aplicar la máscara al mapa de profundidad
    mask_resized = cv2.resize(mask, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked_depth = prediction * mask_resized
    valid_pixels = masked_depth[mask_resized > 0]

    # Guardar mapa de profundidad para revisión
    depth_vis = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    cv2.imwrite(f"{DEPTH_MAPS_SAVE_PATH}depth_map_{frame_id:04d}.jpg", depth_vis)

    if valid_pixels.size == 0:
        return -1  # No hay píxeles válidos
    return np.mean(valid_pixels)


try:
    # ======================
    # PRIMERA ETAPA: detección inicial (sofá)
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

        # Guardar métricas de la primera etapa
        guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected,
                           center_x, center_y, offset_x, offset_y,
                           frame_center_x, frame_center_y)

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
    PROFUNDIDAD_REFERENCIA = None  # Nueva referencia de profundidad inicial
    PROFUNDIDAD_TOLERANCIA = 0.2  # 20% de tolerancia

    etapa2_start = time.time()
    while time.time() - etapa2_start < SEGUNDA_ETAPA_DURACION:
        frame = frame_read.frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = time.time()
        frame_id += 1

        # Copia para dibujar anotaciones (bboxes + etiquetas) de ambos modelos
        annotated_frame = frame.copy()  # BGR

        velocidad_frontal = 0  # evitar errores si no se detecta el
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
                            yaw_velocity = int(np.clip(offset_x / 2, -YAW_SUAVE, YAW_SUAVE))

                    # ======================
                    # TERCERA ETAPA: Ajustar distancia al objeto en base a la profundidad detectada
                    # ======================
                    depth_value = -1
                    depth_error = 0
                    velocidad_frontal = 0

                    if sofa_detected:
                        depth_value = estimate_masked_depth(frame, mask)
                        if depth_value > 0:
                            if PROFUNDIDAD_REFERENCIA is None:
                                PROFUNDIDAD_REFERENCIA = depth_value
                                print(f"[INFO] Profundidad referencia inicial: {PROFUNDIDAD_REFERENCIA:.2f}")
                            else:
                                diferencia_relativa = (depth_value - PROFUNDIDAD_REFERENCIA) / PROFUNDIDAD_REFERENCIA

                                if abs(diferencia_relativa) > PROFUNDIDAD_TOLERANCIA:
                                    depth_error = diferencia_relativa
                                    velocidad_frontal = int(np.clip(depth_error * 10, -20, 20))  # Factor de corrección
                                    print(
                                        f"[CORRECCIÓN] Desviación relativa: {diferencia_relativa:.2f} → Corrigiendo con {velocidad_frontal}")
                                else:
                                    print("[INFO] Profundidad estable dentro de tolerancia")
                                    velocidad_frontal = 0
                                    depth_error = 0
                        else:
                            depth_value = -1
                            depth_error = 0
                            velocidad_frontal = 0
                    else:
                        depth_value = -1
                        depth_error = 0
                        velocidad_frontal = 0
        print("DEPTH VALUE DE : ", depth_value)
        print("DEPTH ERROR DE : ", depth_error)
        tello.send_rc_control(-VELOCIDAD_LATERAL, velocidad_frontal, 0, yaw_velocity)

        # Guardar métricas de posición / profundidad
        guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected,
                           center_x, center_y, offset_x, offset_y,
                           frame_center_x, frame_center_y, depth_value, depth_error)

        # ======================
        # CUARTA ETAPA: detección de partes y desperfectos sobre la segmentación
        # ======================
        if sofa_detected:
            # Preparar imagen enmascarada (solo la región segmentada)
            # … dentro de la cuarta etapa, justo antes del bitwise_and:
            # Asegurar dtype uint8
            mask_output = mask_output.astype(np.uint8)

            # Obtener ancho y alto del frame actual
            h_frame, w_frame = frame_rgb.shape[:2]

            # Si las dimensiones no coinciden, redimensionar la máscara
            if mask_output.shape != (h_frame, w_frame):
                # cv2.resize usa (ancho, alto)
                mask_output = cv2.resize(mask_output, (w_frame, h_frame), interpolation=cv2.INTER_NEAREST)

            # Asegurar que sea monocanal (shape == (h, w))
            if mask_output.ndim == 3:
                mask_output = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)

            masked_frame = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask_output)

            # Convertir masked_frame a formato esperado por YOLO (RGB) ya lo tenemos en RGB
            # Detectar PARTES DE COCHE
            results_parts = model_parts(masked_frame)[0]
            for box, cls_idx, conf in zip(results_parts.boxes.xyxy, results_parts.boxes.cls, results_parts.boxes.conf):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                area = width * height
                class_name = model_parts.names[int(cls_idx)]
                timestamp_form = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                detection_metrics.append({
                    "frame": frame_id,
                    "timestamp": timestamp_form,
                    "modelo": "parts",
                    "clase": class_name,
                    "confianza": float(conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "area": area
                })
                # Dibujar bbox y etiqueta en annotated_frame (BGR)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # verde
                cv2.putText(annotated_frame, f"Part: {class_name} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Detectar DES PERFECTOS
            results_defects = model_defects(masked_frame)[0]
            for box, cls_idx, conf in zip(results_defects.boxes.xyxy, results_defects.boxes.cls, results_defects.boxes.conf):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                area = width * height
                class_name = model_defects.names[int(cls_idx)]
                timestamp_form = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                detection_metrics.append({
                    "frame": frame_id,
                    "timestamp": timestamp_form,
                    "modelo": "defects",
                    "clase": class_name,
                    "confianza": float(conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "area": area
                })
                # Dibujar bbox y etiqueta en annotated_frame (BGR)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rojo
                cv2.putText(annotated_frame, f"Defect: {class_name} {conf:.2f}",
                            (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Guardar el frame anotado con todas las bboxes de ambos modelos
            cv2.imwrite(f"{MODELS_DETECTION_PATH}det_{frame_id:04d}.jpg", annotated_frame)

        time.sleep(0.1)

    print("[INFO] Segunda etapa completada. Aterrizando...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()

except KeyboardInterrupt:
    print("[INFO] Interrupción manual. Aterrizando...")
    tello.land()

finally:
    # Guardar métricas de profundidad/posición
    df = pd.DataFrame(metrics)
    df.to_csv(METRICS_SAVE_PATH, index=False)

    # ======================
    # GUARDAR MÉTRICAS DE DETECCIONES
    # ======================
    detections_csv_path = "4.Detections/detection_metrics.csv"
    df_det = pd.DataFrame(detection_metrics)
    df_det.to_csv(detections_csv_path, index=False)
    print(f"[INFO] Métricas de detecciones guardadas en {detections_csv_path}")

    tello.streamoff()
    tello.end()
    print("[INFO] Vuelo finalizado. Datos guardados.")
