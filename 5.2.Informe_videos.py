import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from fpdf import FPDF
from datetime import datetime

# CONFIGURACIÓN
VIDEO_PATH = r"C:\Users\USUARIO\TFG\videos_coches\Video.mp4"  # Ruta al video local
FRAME_SAVE_PATH = "5.2.Informe_videos/frames/"
METRICS_SAVE_PATH = "5.2.Informe_videos/metrics/"
MODELS_DETECTION_PATH = "5.2.Informe_videos/detections/"
REPORT_PATH = "5.2.Informe_videos/informe_video.pdf"

MODEL_SEG_PATH     = "yolov8n-seg.pt"
MODEL_PARTS_PATH   = r"C:\Users\USUARIO\TFG\modelos_pruebas\car_PARTS_model2\weights\best.pt"
MODEL_DEFECTS_PATH = r"C:\Users\USUARIO\TFG\modelos_pruebas\car_dd_yolov8n3\weights\best.pt"
CONF_THRESHOLD     = 0.5
TARGET_CLASS       = "car"

# Crear carpetas de salida
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
os.makedirs(METRICS_SAVE_PATH, exist_ok=True)
os.makedirs(MODELS_DETECTION_PATH, exist_ok=True)

# Cargar modelos
model_seg     = YOLO(MODEL_SEG_PATH)
model_parts   = YOLO(MODEL_PARTS_PATH)
model_defects = YOLO(MODEL_DEFECTS_PATH)

# Iniciar medición de tiempo
t0 = time.time()

# Abrir vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"No se pudo abrir el vídeo: {VIDEO_PATH}")

frame_id = 0
detection_metrics = []

# Procesamiento frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== SEGMENTACIÓN DE COCHE =====
    res = model_seg(frame_rgb)[0]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if res.masks is None:
        continue
    for seg, cls in zip(res.masks.data, res.boxes.cls):
        if model_seg.names[int(cls)] == TARGET_CLASS:
            seg_mask = (seg.cpu().numpy().astype(np.uint8) * 255)
            seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, seg_mask)
            break
    if mask.sum() == 0:
        continue

    # Preparar imágenes anotadas
    masked    = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
    annotated = frame.copy()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detección de partes
    res_p = model_parts(masked)[0]
    for box, cls_i, conf in zip(res_p.boxes.xyxy, res_p.boxes.cls, res_p.boxes.conf):
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        area_px = (x2 - x1) * (y2 - y1)
        name = model_parts.names[int(cls_i)]
        detection_metrics.append({
            "frame": frame_id,
            "timestamp": timestamp_str,
            "modelo": "parts",
            "clase": name,
            "confianza": float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area_px": area_px
        })
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"Part: {name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Detección de desperfectos
    res_d = model_defects(masked)[0]
    for box, cls_i, conf in zip(res_d.boxes.xyxy, res_d.boxes.cls, res_d.boxes.conf):
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        area_px = (x2 - x1) * (y2 - y1)
        name = model_defects.names[int(cls_i)]
        detection_metrics.append({
            "frame": frame_id,
            "timestamp": timestamp_str,
            "modelo": "defects",
            "clase": name,
            "confianza": float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area_px": area_px
        })
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"Defect: {name} {conf:.2f}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Guardar frame anotado
    out_img = os.path.join(MODELS_DETECTION_PATH, f"det_{frame_id:04d}.jpg")
    cv2.imwrite(out_img, annotated)

# Liberar recursos
cap.release()

# Guardar métricas
df = pd.DataFrame(detection_metrics)
metrics_csv = os.path.join(METRICS_SAVE_PATH, "detection_metrics.csv")
df.to_csv(metrics_csv, index=False)

# Asociar desperfectos a partes
parts_df   = df[df["modelo"] == "parts"]
defects_df = df[df["modelo"] == "defects"]
maps = []
for _, defect in defects_df.iterrows():
    f = defect["frame"]
    dx1, dy1, dx2, dy2 = defect[["x1","y1","x2","y2"]]
    da = defect["area_px"]
    match_p = "N/A"
    best_iou = 0
    for _, part in parts_df[parts_df["frame"] == f].iterrows():
        px1, py1, px2, py2 = part[["x1","y1","x2","y2"]]
        ix1, iy1 = max(dx1, px1), max(dy1, py1)
        ix2, iy2 = min(dx2, px2), min(dy2, py2)
        if ix2>ix1 and iy2>iy1:
            inter = (ix2-ix1)*(iy2-iy1)
            union = da + part["area_px"] - inter
            iou = inter/union if union>0 else 0
            if iou > best_iou:
                best_iou, match_p = iou, part["clase"]
    maps.append({
        "defect_class": defect["clase"],
        "part_class": match_p,
        "confidence": defect["confianza"],
        "area_px": da,
        "frame": f
    })
map_df    = pd.DataFrame(maps)
unique_df = map_df.drop_duplicates(subset=["defect_class","part_class"])

# Estadísticas y tiempo
total_time = time.time() - t0
minutes    = int(total_time // 60)
seconds    = int(total_time % 60)
num_frames = df["frame"].nunique()
unique_pairs = unique_df.shape[0]
now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Generar PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial","B",16)
pdf.cell(0,10,"Informe Detección Vídeo",ln=1)
pdf.set_font("Arial","",12)
pdf.cell(0,8,f"Fecha: {now}",ln=1)
pdf.cell(0,8,f"Tiempo procesamiento: {minutes} min {seconds} s",ln=1)
pdf.cell(0,8,f"Frames procesados: {num_frames}",ln=1)
pdf.cell(0,8,f"Pares únicos defecto-parte: {unique_pairs}",ln=1)

pdf.ln(10)
# Tabla de resultados únicos
pdf.set_font("Arial","B",12)
th = pdf.font_size + 2
pdf.cell(40,th,"Desperfecto",1)
pdf.cell(40,th,"Parte",1)
pdf.cell(30,th,"Confianza",1)
pdf.cell(40,th,"Área (px2)",1)
pdf.ln(th)
pdf.set_font("Arial","",12)
for _, r in unique_df.iterrows():
    pdf.cell(40,th,str(r["defect_class"]),1)
    pdf.cell(40,th,str(r["part_class"]),1)
    pdf.cell(30,th,f"{r['confidence']:.2f}",1)
    pdf.cell(40,th,str(int(r['area_px'])),1)
    pdf.ln(th)

# Página de imágenes con encabezado
pdf.add_page()
for _, row in unique_df.iterrows():
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,f"Frame {int(row['frame'])}: Desperfecto {row['defect_class']} - Parte {row['part_class']}",ln=1)
    img_path = os.path.join(MODELS_DETECTION_PATH, f"det_{int(row['frame']):04d}.jpg")
    if os.path.exists(img_path):
        pdf.image(img_path, w=100)
        pdf.ln(5)

# Guardar reporte
pdf.output(REPORT_PATH)
print(f"Informe generado: {REPORT_PATH} (Tiempo total: {minutes} min {seconds} s)")
