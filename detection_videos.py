#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ver_video_detecciones.py

Script para:
  1) Cargar dos modelos YOLOv8n (.pt) entrenados:
       - detección de partes (parts_best.pt)
       - detección de desperfectos (damages_best.pt)
  2) Abrir un vídeo desde disco.
  3) Por cada fotograma, realizar inferencia con ambos modelos.
     - Dibujar los bounding‐boxes de "parts" en azul.
     - Dibujar los bounding‐boxes de "damages" en rojo.
  4) Mostrar en pantalla el vídeo con las detecciones superpuestas.
  5) (Opcional) Registrar en un CSV las métricas de cada detección.

Requisitos:
  - Python 3.7+
  - ultralytics (para YOLOv8)
  - opencv-python
  - pandas

Instalación de dependencias:
    pip install ultralytics opencv-python pandas

Uso:
    python detection_videos.py

"""

import os
import cv2
import pandas as pd
from ultralytics import YOLO
import datetime

def cargar_modelo_yolov8(path_peso: str) -> YOLO:
    """
    Carga un modelo YOLOv8n entrenado (archivo .pt) usando la librería ultralytics.
    """
    if not os.path.isfile(path_peso):
        raise FileNotFoundError(f"No se encontró el fichero de pesos: {path_peso}")
    modelo = YOLO(path_peso)
    return modelo

def dibujar_detecciones(
    frame,
    resultados,
    color_bbox: tuple,
    etiqueta_modelo: str,
    font=cv2.FONT_HERSHEY_SIMPLEX
):
    """
    Dibuja sobre el frame las cajas de detección del objeto `resultados` (YOLOv8),
    usando el color indicado. También pone la etiqueta de clase y confidencia.

    - frame: imagen BGR donde pintar.
    - resultados: objeto de tipo ultralytics.engine.results.Results (lista con un elemento).
    - color_bbox: tupla BGR, p.ej. (255, 0, 0) para azul.
    - etiqueta_modelo: nombre corto para prefijar el texto, p.ej. "PART" o "DAMA".
    """
    r = resultados[0]  # en YOLOv8, predict() devuelve lista; queremos el primer item (arrays de boxes)
    if r.boxes is None or len(r.boxes) == 0:
        return

    # Cada elemento de r.boxes es una instancia de ultralytics.yolo.engine.results.Boxes
    for box in r.boxes:
        # box.xyxy: tensor [[x1, y1, x2, y2]]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]

        # Dibujar rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bbox, 2)

        # Preparar texto: "<etiqueta_modelo>:<clase> <conf:0.2f>"
        texto = f"{etiqueta_modelo}:{cls_name} {conf:.2f}"
        # Calcular posición del texto (justo encima de la caja si cabe)
        (text_w, text_h), _ = cv2.getTextSize(texto, font, 0.5, 1)
        y_text = y1 - 5 if y1 - 5 > text_h else y1 + text_h + 5
        cv2.rectangle(frame, (x1, y_text - text_h - 2), (x1 + text_w + 2, y_text), color_bbox, -1)
        cv2.putText(frame, texto, (x1 + 1, y_text - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def procesar_y_mostrar_video(
    ruta_video: str,
    modelo_parts: YOLO,
    modelo_damages: YOLO,
    ruta_salida_csv: str = None,
    guardar_intervalo: int = 1000
):
    """
    Abre el vídeo con OpenCV, itera fotograma a fotograma, aplica inferencia con ambos
    modelos YOLOv8 y pinta las detecciones en pantalla.

    Opcionalmente, guarda métricas de cada detección en un CSV.

    Parámetros:
      - ruta_video: ruta al fichero .mp4
      - modelo_parts: instancia de YOLO cargada para detección de piezas
      - modelo_damages: instancia de YOLO cargada para detección de desperfectos
      - ruta_salida_csv: (opcional) si se proporciona, guarda un CSV con las siguientes columnas:
           frame, timestamp, modelo, clase, confianza, x1, y1, x2, y2, width, height, area
      - guardar_intervalo: cada cuántos fotogramas va volcando al CSV (por lotes)
    """
    if not os.path.isfile(ruta_video):
        raise FileNotFoundError(f"No se encontró el vídeo: {ruta_video}")

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {ruta_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    registros = []
    frame_idx = 0
    batch_counter = 0

    print(f"[INFO] Abriendo vídeo:")
    print(f"       Ruta: {ruta_video}")
    print(f"       FPS estimados: {fps:.2f}")
    print(f"       Total fotogramas: {total_frames}")
    print(f"[INFO] Presiona 'q' para salir antes de que termine el vídeo.")

    ventana_nombre = "VIDEO - Detecciones"
    cv2.namedWindow(ventana_nombre, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana_nombre, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # final del vídeo

        # Timestamp en segundos y formato legible:
        timestamp_seg = frame_idx / fps
        timestamp_form = str(datetime.timedelta(seconds=timestamp_seg))

        # Convertir BGR→RGB solo si fuera necesario internamente,
        # pero YOLOv8 acepta directamente la imagen BGR de OpenCV.
        # Inferencia MODELO PARTS
        resultados_parts = modelo_parts.predict(frame, verbose=False)[0]
        # Inferencia MODELO DAMAGES
        resultados_damages = modelo_damages.predict(frame, verbose=False)[0]

        # Dibujar todas las detecciones de PARTS en azul (BGR=(255,0,0))
        dibujar_detecciones(
            frame,
            [resultados_parts],
            color_bbox=(255, 0, 0),
            etiqueta_modelo="PART"
        )

        # Dibujar todas las detecciones de DAMAGES en rojo (BGR=(0,0,255))
        dibujar_detecciones(
            frame,
            [resultados_damages],
            color_bbox=(0, 0, 255),
            etiqueta_modelo="DAMA"
        )

        # Si se quiere guardar métricas en CSV:
        if ruta_salida_csv is not None:
            # Recolectar métricas de PARTS
            for box in resultados_parts.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = resultados_parts.names[cls_id]
                width = x2 - x1
                height = y2 - y1
                area = width * height

                registros.append({
                    "frame": frame_idx,
                    "timestamp": timestamp_form,
                    "modelo": "parts",
                    "clase": cls_name,
                    "confianza": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "area": area
                })

            # Recolectar métricas de DAMAGES
            for box in resultados_damages.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = resultados_damages.names[cls_id]
                width = x2 - x1
                height = y2 - y1
                area = width * height

                registros.append({
                    "frame": frame_idx,
                    "timestamp": timestamp_form,
                    "modelo": "damages",
                    "clase": cls_name,
                    "confianza": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "area": area
                })

            batch_counter += 1
            # Volcar cada cierto número de fotogramas al CSV para no saturar memoria
            if batch_counter >= guardar_intervalo:
                df_parcial = pd.DataFrame(registros)
                modo = 'a' if os.path.isfile(ruta_salida_csv) else 'w'
                header = not os.path.isfile(ruta_salida_csv)
                df_parcial.to_csv(ruta_salida_csv, mode=modo, index=False, header=header)
                registros = []
                batch_counter = 0
                print(f"[INFO] Guardados {guardar_intervalo} fotogramas en '{ruta_salida_csv}'")

        # Mostrar el frame con detecciones
        cv2.imshow(ventana_nombre, frame)
        delay_ms = int(1000 / fps)  # → 33 ms aprox.
        cv2.waitKey(delay_ms)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Salida solicitada por el usuario (tecla 'q').")
            break

        frame_idx += 1

    # Al terminar el bucle, volcamos las detecciones restantes al CSV
    if ruta_salida_csv is not None and registros:
        df_final = pd.DataFrame(registros)
        modo = 'a' if os.path.isfile(ruta_salida_csv) else 'w'
        header = not os.path.isfile(ruta_salida_csv)
        df_final.to_csv(ruta_salida_csv, mode=modo, index=False, header=header)
        print(f"[INFO] Guardados los últimos {len(registros)} registros en '{ruta_salida_csv}'")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Procesamiento y visualización finalizados.")

if __name__ == "__main__":
    # ---------- Configuración de rutas ----------
    RUTA_VIDEO = r"C:\Users\USUARIO\TFG\videos_coches\WIN_20250103_11_19_37_Pro.mp4"
    RUTA_MODEL_PARTS   = r"C:\Users\USUARIO\TFG\modelos_pruebas\parts_best.pt"
    RUTA_MODEL_DAMAGES = r"C:\Users\USUARIO\TFG\modelos_pruebas\damages_best.pt"
    # Si no quieres guardar CSV, pon RUTA_SALIDA_CSV = None
    RUTA_SALIDA_CSV = r"C:\Users\USUARIO\TFG\modelos_pruebas\resultados_detecciones.csv"

    # ---------- Carga de modelos ----------
    print("[INFO] Cargando modelo YOLOv8 PARTS desde:", RUTA_MODEL_PARTS)
    modelo_parts = cargar_modelo_yolov8(RUTA_MODEL_PARTS)

    print("[INFO] Cargando modelo YOLOv8 DAMAGES desde:", RUTA_MODEL_DAMAGES)
    modelo_damages = cargar_modelo_yolov8(RUTA_MODEL_DAMAGES)

    # ---------- Procesar y mostrar vídeo ----------
    procesar_y_mostrar_video(
        ruta_video=RUTA_VIDEO,
        modelo_parts=modelo_parts,
        modelo_damages=modelo_damages,
        ruta_salida_csv=RUTA_SALIDA_CSV,
        guardar_intervalo=500  # cada 500 fotogramas se vuelca el batch a disco
    )
