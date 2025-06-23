# Proyecto de Inspección Autónoma de Vehículos con Dron

Este repositorio contiene el desarrollo completo de un sistema autónomo para la inspección de daños en vehículos mediante un dron DJI Tello, apoyado únicamente en una cámara monocular y técnicas de visión por computador e inteligencia artificial.

El objetivo principal es realizar un vuelo orbital autónomo alrededor de un coche, manteniendo su centrado y distancia constante mediante segmentación semántica (YOLOv8-seg) y estimación de profundidad (MiDaS), para finalmente detectar partes y desperfectos en el vehículo e integrar los resultados en un informe pericial automático.

---

## 📁 Estructura del repositorio

Cada carpeta corresponde a una fase o experimento individual del desarrollo del proyecto:

### `1.inicio/`
Primeras pruebas de detección y segmentación de objetos con el dron en vuelo estático. Incluye:
- `frames/`: Imágenes capturadas
- `masks/`: Segmentaciones
- `detections/`: Resultados de detección

---

### `2.movement/`
Implementación del movimiento lateral del dron, manteniendo el objeto centrado en el encuadre. Incluye resultados similares al anterior.

---

### `3.MiDaS/`
Implementación del modelo MiDaS para estimar profundidad a partir de una imagen monocular:
- `3.1.depth_estimation/`: Resultados experimentales ordenados por fecha
- `3.2.Midas_Extra/`: Utilidades y pruebas adicionales
- Carpetas globales para `depth_maps`, `frames`, `masks` y `detections`

---

### `4.Detections/`
Aplicación de modelos entrenados para la **detección de partes** y **desperfectos** sobre las segmentaciones del coche:
- `models_detections/`: Resultados por modelo (YOLOv8n, YOLOv8n-seg...)
- `frames/`, `masks/`, `detections/`, `depth_maps/`: Resultados organizados

---

### `5.InformeFinal/`, `5.3.InformeFinal/`, etc.
Generación automatizada del informe pericial con todas las métricas e imágenes relevantes. Contiene:
- `5.1.informes/`: Resultados ordenados por tipo
- `frames/`, `masks/`, `depth_maps/`, `detections/`: Datos para el informe
- `models_detections/`: Resultados con detecciones específicas para el informe final

> Existen varias versiones (`- copia`, `5.2`, `5.3`) debido a iteraciones y mejoras del sistema.

---

### Otras carpetas auxiliares
- `capturas_originales/`: Frames sin procesar
- `profundidades_midas/`: Exportaciones manuales de mapas de profundidad

---

## 🔧 Tecnologías y modelos utilizados

- **Dron**: DJI Tello
- **Visión por computador**: OpenCV
- **Detección y segmentación**: YOLOv8n y YOLOv8n-seg (Ultralytics)
- **Estimación de profundidad**: MiDaS (torchvision + timm)
- **Procesamiento**: Python, Pandas, PIL
- **Visualización y PDF**: FPDF, OpenCV
- **Entrenamiento de modelos**: PyTorch y Ultralytics

---

## 📑 Resultados

El sistema permite realizar de forma autónoma:
- La detección y seguimiento del vehículo en vuelo
- La estimación de profundidad para mantener la distancia al objeto
- La detección de partes y desperfectos del coche
- La generación de un informe final estructurado con métricas y capturas clave

---

## 📷 Ejemplos visuales

Se pueden reutilizar las imágenes incluidas en la memoria del proyecto para representar:
- El pipeline completo del sistema
- Ejemplos de detecciones correctas y fallidas
- Comparativa entre profundidad estimada y real

---

## 📌 Nota

Este repositorio refleja un proyecto académico y ha sido estructurado para recoger las distintas fases experimentales del desarrollo. La organización por carpetas busca documentar y separar claramente cada etapa sin sobreescribir resultados previos.

---

## 📫 Contacto

Proyecto desarrollado por María Torres como parte del Trabajo de Fin de Grado en Ingeniería y Sistemas de Datos.  
Universidad Politécnica de Madrid, 2025.


