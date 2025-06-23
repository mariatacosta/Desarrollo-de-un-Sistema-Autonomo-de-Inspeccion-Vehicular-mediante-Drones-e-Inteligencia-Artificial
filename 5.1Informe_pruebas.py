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
from fpdf import FPDF
from datetime import datetime

# ======================
# CONFIGURACIÓN
# ======================
TARGET_CLASS = "couch"
FRAME_SAVE_PATH        = "5.InformeFinal/5.1.informes/frames/"
MASK_SAVE_PATH         = "5.InformeFinal/5.1.informes/masks/"
DETECTION_SAVE_PATH    = "5.InformeFinal/5.1.informes/detections/"
METRICS_SAVE_PATH      = "5.InformeFinal/5.1.informes/metrics.csv"
DEPTH_MAPS_SAVE_PATH   = "5.InformeFinal/5.1.informes/depth_maps/"
MODELS_DETECTION_PATH  = "5.InformeFinal/5.1.informes/models_detections/"

MODEL_PARTS_PATH   = r"C:\Users\USUARIO\TFG\modelos_pruebas\parts_best.pt"
MODEL_DEFECTS_PATH = r"C:\Users\USUARIO\TFG\modelos_pruebas\damages_best.pt"
CONF_THRESHOLD = 0.5

NO_DETECTION_TIMEOUT   = 10
VELOCIDAD_LATERAL      = 13
YAW_SUAVE              = 25
HUMBRAL_OFFSET         = 60
SEGUNDA_ETAPA_DURACION = 20

PROFUNDIDAD_MIN = 2.3
PROFUNDIDAD_MAX = 2.7
PROFUNDIDAD_OBJETIVO = (PROFUNDIDAD_MIN + PROFUNDIDAD_MAX) / 2  # 2.5

# Crear carpetas si no existen
os.makedirs(FRAME_SAVE_PATH,       exist_ok=True)
os.makedirs(MASK_SAVE_PATH,        exist_ok=True)
os.makedirs(DETECTION_SAVE_PATH,   exist_ok=True)
os.makedirs(DEPTH_MAPS_SAVE_PATH,  exist_ok=True)
os.makedirs(MODELS_DETECTION_PATH, exist_ok=True)

# ======================
# INICIALIZACIÓN
# ======================
model = YOLO("yolov8n-seg.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("isl-org/MiDaS", "DPT_LeViT_224", trust_repo=True)
midas.to(device).eval()

midas_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])
print("✅ MiDaS cargado correctamente.")

model_parts   = YOLO(MODEL_PARTS_PATH)
model_defects = YOLO(MODEL_DEFECTS_PATH)
print("✅ Modelos de partes y desperfectos cargados correctamente.")

tello = Tello()
tello.connect()
tello.streamon()
print(f"[INFO] Batería inicial: {tello.get_battery()}%")

tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)
print("[INFO] Dron despegado")

frame_read      = tello.get_frame_read()
frame           = frame_read.frame
frame_height, frame_width = frame.shape[:2]
frame_center_x  = frame_width  // 2
frame_center_y  = frame_height // 2

metrics            = []  # posición/profundidad
detection_metrics  = []  # partes y desperfectos

frame_id            = 0
start_time          = time.time()
last_detection_time = start_time
PROFUNDIDAD_REFERENCIA = None

# --------------------------------------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------------------------------------
def guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected,
                       center_x, center_y, offset_x, offset_y,
                       frame_center_x, frame_center_y,
                       depth_value=None, depth_error=None):
    metrics.append({
        "frame_id"      : frame_id,
        "timestamp"     : time.time(),
        "sofa_detected" : int(sofa_detected),
        "centroid_x"    : center_x if sofa_detected else -1,
        "centroid_y"    : center_y if sofa_detected else -1,
        "frame_center_x": frame_center_x,
        "frame_center_y": frame_center_y,
        "offset_x"      : offset_x if sofa_detected else -1,
        "offset_y"      : offset_y if sofa_detected else -1,
        "centrado"      : int(abs(offset_x) < HUMBRAL_OFFSET if sofa_detected else 0),
        "depth_value"   : depth_value if depth_value is not None else -1,
        "depth_error"   : depth_error if depth_error is not None else -1
    })

    cv2.imwrite(f"{FRAME_SAVE_PATH}frame_{frame_id:04d}.jpg", frame_rgb)
    cv2.imwrite(f"{MASK_SAVE_PATH}mask_{frame_id:04d}.png",  mask_output)

    if sofa_detected:
        mask_color = cv2.merge([np.zeros_like(mask_output),
                                np.zeros_like(mask_output),
                                mask_output])
        mask_color = cv2.resize(mask_color, frame_rgb.shape[1::-1])
        overlay    = cv2.addWeighted(frame_rgb, 1.0, mask_color.astype(np.uint8), 0.5, 0)
        cv2.imwrite(f"{DETECTION_SAVE_PATH}overlay_{frame_id:04d}.jpg", overlay)

def estimate_masked_depth(frame, mask):
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    img_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    input_t = midas_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = midas(input_t)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=frame.shape[:2],
            mode="bicubic", align_corners=False).squeeze().cpu().numpy()

    mask_r = cv2.resize(mask, pred.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    masked_depth = pred * mask_r
    valid = masked_depth[mask_r > 0]

    depth_vis = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"{DEPTH_MAPS_SAVE_PATH}depth_map_{frame_id:04d}.jpg", depth_vis)

    return np.mean(valid) if valid.size else -1

# ==================================================================================
# PRIMERA ETAPA: búsqueda inicial del sofá
# ==================================================================================
try:
    while True:
        frame      = frame_read.frame
        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_id  += 1
        tello.send_rc_control(0, 0, 0, 0)

        results       = model(frame_rgb)[0]
        sofa_detected = False
        mask_output   = np.zeros((frame_height, frame_width), dtype=np.uint8)

        center_x = center_y = offset_x = offset_y = None

        if results.masks is not None:
            for seg, cls in zip(results.masks.data, results.boxes.cls):
                if model.names[int(cls)] == TARGET_CLASS:
                    sofa_detected      = True
                    last_detection_time = time.time()
                    mask_output        = (seg.cpu().numpy().astype(np.uint8) * 255)
                    mask_output        = cv2.resize(mask_output, frame.shape[1::-1])

                    M = cv2.moments(mask_output)
                    if M["m00"] > 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        offset_x = center_x - frame_center_x
                        offset_y = center_y - frame_center_y
                    break

        guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected,
                           center_x, center_y, offset_x, offset_y,
                           frame_center_x, frame_center_y)

        if sofa_detected:
            print(f"[INFO] '{TARGET_CLASS}' detectado en frame {frame_id}. ✅ PASANDO A LA SEGUNDA ETAPA…")
            break

        if time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
            print("[WARN] No se detectó el sofá dentro del tiempo límite. Abortando.")
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            raise SystemExit

    # ==================================================================================
    # SEGUNDA ETAPA
    # ==================================================================================
    PROFUNDIDAD_REFERENCIA = None
    PROFUNDIDAD_TOLERANCIA = 0.2
    etapa2_start = time.time()

    while time.time() - etapa2_start < SEGUNDA_ETAPA_DURACION:
        frame     = frame_read.frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_id += 1

        annotated_frame = frame.copy()
        results         = model(frame_rgb)[0]
        sofa_detected   = False
        offset_x        = 0
        yaw_velocity    = 0
        velocidad_frontal = 0

        mask_output = np.zeros((frame_height, frame_width), dtype=np.uint8)
        center_x = center_y = offset_y = -1
        depth_value = depth_error = -1

        if results.masks is not None:
            for seg, cls in zip(results.masks.data, results.boxes.cls):
                if model.names[int(cls)] == TARGET_CLASS:
                    sofa_detected   = True
                    mask_output     = (seg.cpu().numpy().astype(np.uint8) * 255)
                    mask_output     = cv2.resize(mask_output, frame.shape[1::-1])

                    M = cv2.moments(mask_output)
                    if M["m00"] > 0:
                        center_x  = int(M["m10"] / M["m00"])
                        center_y  = int(M["m01"] / M["m00"])
                        offset_x  = center_x - frame_center_x
                        if abs(offset_x) > HUMBRAL_OFFSET:
                            yaw_velocity = int(np.clip(offset_x / 2, -YAW_SUAVE, YAW_SUAVE))

                    depth_value = estimate_masked_depth(frame, mask_output)
                    if depth_value > 0:
                        if PROFUNDIDAD_REFERENCIA is None:
                            PROFUNDIDAD_REFERENCIA = depth_value
                            print(f"[INFO] Profundidad de referencia: {PROFUNDIDAD_REFERENCIA:.2f} m")
                        else:
                            rel_error = (depth_value - PROFUNDIDAD_REFERENCIA) / PROFUNDIDAD_REFERENCIA
                            if abs(rel_error) > PROFUNDIDAD_TOLERANCIA:
                                depth_error = rel_error
                                velocidad_frontal = int(np.clip(rel_error * 10, -20, 20))
                            else:
                                depth_error = 0
                                velocidad_frontal = 0
                    break  # solo primer sofá

        tello.send_rc_control(-VELOCIDAD_LATERAL, velocidad_frontal, 0, yaw_velocity)

        guardar_resultados(metrics, frame_id, frame_rgb, mask_output, sofa_detected,
                           center_x, center_y, offset_x, offset_y,
                           frame_center_x, frame_center_y, depth_value, depth_error)

        # --------------------------------------------------------------------------------
        # CUARTA ETAPA: detección de partes y desperfectos
        # --------------------------------------------------------------------------------
        if sofa_detected:
            mask_output = mask_output.astype(np.uint8)
            h, w = frame.shape[:2]
            if mask_output.shape != (h, w):
                mask_output = cv2.resize(mask_output, (w, h), interpolation=cv2.INTER_NEAREST)
            if mask_output.ndim == 3:
                mask_output = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)

            masked_frame = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask_output)

            # PARTES
            for box, cls_i, conf in zip(*model_parts(masked_frame)[0].boxes.xyxy_cls_conf):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                detection_metrics.append({
                    "frame"   : frame_id,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    "modelo"  : "parts",
                    "clase"   : model_parts.names[int(cls_i)],
                    "confianza": float(conf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1, "area": area
                })
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Part:{model_parts.names[int(cls_i)]} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # DESPERFECTOS
            for box, cls_i, conf in zip(*model_defects(masked_frame)[0].boxes.xyxy_cls_conf):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                detection_metrics.append({
                    "frame"   : frame_id,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    "modelo"  : "defects",
                    "clase"   : model_defects.names[int(cls_i)],
                    "confianza": float(conf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1, "area": area
                })
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Def:{model_defects.names[int(cls_i)]} {conf:.2f}",
                            (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imwrite(f"{MODELS_DETECTION_PATH}det_{frame_id:04d}.jpg", annotated_frame)

        time.sleep(0.1)

    print("[INFO] Segunda etapa completada. Aterrizando…")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()

except KeyboardInterrupt:
    print("[INFO] Interrupción manual. Aterrizando…")
    tello.land()

finally:
    # =========================================================================
    # GUARDAR MÉTRICAS
    # =========================================================================
    pd.DataFrame(metrics).to_csv(METRICS_SAVE_PATH, index=False)

    detections_csv_path = "5.InformeFinal/5.1.informes/detection_metrics.csv"
    pd.DataFrame(detection_metrics).to_csv(detections_csv_path, index=False)
    print(f"[INFO] Métricas de detecciones guardadas en {detections_csv_path}")

    # =========================================================================
    # QUINTA ETAPA: CREAR INFORME PDF
    # =========================================================================
    df_metrics = pd.read_csv(METRICS_SAVE_PATH)
    df_det     = pd.read_csv(detections_csv_path)

    # ---------- AGRUPAR PARTES Y DESPERFECTOS --------------------------------
    parts_df        = df_det[df_det["modelo"] == "parts"]
    raw_defects_df  = df_det[df_det["modelo"] == "defects"]

    # Eliminar duplicados exactos (mismo frame y misma bbox)
    defects_df = (raw_defects_df
                  .drop_duplicates(subset=["frame", "x1", "y1", "x2", "y2"])  # ### MOD <–
                  .reset_index(drop=True))                                    # ### MOD <–

    # Estadísticas
    creation_date  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_frames     = int(df_metrics[df_metrics["sofa_detected"] == 1]["frame_id"].nunique())
    unique_parts   = parts_df["clase"].nunique()
    unique_defects = defects_df.shape[0]             # ### MOD <– (instancias únicas)

    calibration_value = PROFUNDIDAD_REFERENCIA if PROFUNDIDAD_REFERENCIA is not None else -1

    # ---------- RELACIONAR DESPERFECTOS CON PARTES ---------------------------
    mappings    = []
    cal_cm_px   = 0.1
    for _, defect in defects_df.iterrows():          # ### MOD <– usamos defects_df
        frame_idx                     = defect["frame"]
        dx1, dy1, dx2, dy2, d_area    = defect[["x1", "y1", "x2", "y2", "area"]]
        matched_part, max_iou         = "N/A", 0

        for _, part in parts_df[parts_df["frame"] == frame_idx].iterrows():
            px1, py1, px2, py2 = part[["x1", "y1", "x2", "y2"]]
            ix1, iy1 = max(dx1, px1), max(dy1, py1)
            ix2, iy2 = min(dx2, px2), min(dy2, py2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = d_area + part["area"] - inter
                iou   = inter / union if union else 0
                if iou > max_iou:
                    max_iou, matched_part = iou, part["clase"]

        mappings.append({
            "defect_class": defect["clase"],
            "part_class"  : matched_part,
            "confidence"  : defect["confianza"],
            "size_cm2"    : d_area * (cal_cm_px ** 2),
            "frame"       : frame_idx
        })

    map_df = pd.DataFrame(mappings)

    # ---------- GENERAR PDF --------------------------------------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Informe de Detección de Desperfectos", ln=1)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Fecha de creación: {creation_date}", ln=1)
    pdf.cell(0, 8, f"Número de frames analizados: {num_frames}", ln=1)
    pdf.cell(0, 8, f"Número de partes de coche únicas: {unique_parts}", ln=1)
    pdf.cell(0, 8, f"Número total de desperfectos distintos: {unique_defects}", ln=1)  # ### MOD <–
    pdf.cell(0, 8, f"Profundidad de referencia (m): {calibration_value:.2f}", ln=1)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    th = pdf.font_size + 2
    headers = ["Desperfecto", "Parte", "Confianza", "Tamaño (cm²)"]
    for h in headers:
        pdf.cell(40 if h != "Confianza" else 30, th, h, 1)
    pdf.ln(th)

    pdf.set_font("Arial", "", 12)
    for _, row in map_df.iterrows():
        pdf.cell(40, th, str(row["defect_class"]), 1)
        pdf.cell(40, th, str(row["part_class"]),   1)
        pdf.cell(30, th, f"{row['confidence']:.2f}", 1)
        pdf.cell(40, th, f"{row['size_cm2']:.2f}", 1)
        pdf.ln(th)

    pdf.add_page()
    for frame in map_df["frame"].unique():
        img_path = f"{MODELS_DETECTION_PATH}det_{int(frame):04d}.jpg"
        if os.path.exists(img_path):
            pdf.image(img_path, w=100)
            pdf.ln(5)

    report_path = "5.InformeFinal/5.1.informes/informe_deteccion.pdf"
    pdf.output(report_path)
    print(f"[INFO] Informe PDF generado en {report_path}")

    tello.streamoff()
    tello.end()
    print("[INFO] Vuelo finalizado. Datos guardados.")
