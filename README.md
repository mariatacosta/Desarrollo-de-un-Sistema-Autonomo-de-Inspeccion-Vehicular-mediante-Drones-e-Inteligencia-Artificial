# Proyecto de Inspección Autónoma de Vehículos con Dron

Este repositorio contiene el desarrollo completo de un sistema autónomo para la inspección de daños en vehículos mediante un dron DJI Tello. El sistema se basa exclusivamente en visión por computador utilizando una única cámara monocular, integrando modelos de detección, segmentación y estimación de profundidad para navegar y analizar un vehículo de forma completamente autónoma.

---

## 📁 Estructura del repositorio

Cada carpeta representa una fase independiente del desarrollo del sistema. Se ha seguido una progresión modular para implementar y validar los distintos componentes antes de integrarlos en una versión final.

### `1.inicio/`
Primera fase experimental. El dron se mantiene estático mientras se prueba la detección de objetos y se evalúan los resultados básicos:
- `frames/`: Imágenes capturadas
- `masks/`: Segmentaciones generadas
- `detections/`: Detecciones en bruto

---

### `2.movement/`
Implementación del movimiento lateral autónomo del dron, manteniendo el objeto centrado mediante segmentación semántica:
- Se prueban ajustes suaves de orientación y traslación
- Estructura similar para guardar `frames/`, `masks/` y `detections/`

---

### `3.MiDaS/`
Integración del modelo **MiDaS** para estimar la distancia al objeto mediante profundidad monocular:
- `depth_maps/`: Mapas de profundidad generados
- Se utiliza la media de profundidad en la segmentación para ajustar la distancia del dron
- Estructura común: `frames/`, `masks/`, `detections/`

---

### `4.Detections/`
Aplicación de modelos entrenados para la **detección de partes del coche** y **desperfectos** sobre las segmentaciones:
- `models_detections/`: Resultados por modelo (YOLOv8n, etc.)
- Organización por tipo de datos: `frames/`, `masks/`, `detections/`, `depth_maps/`

---

### `5.2.Informe_videos/`
Generación de vídeos experimentales y recolección de métricas durante vuelos de prueba:
- `metrics/`: CSV con métricas temporales del experimento
- Detecciones en vídeo como forma preliminar de informe visual

---

### `5.3.InformeFinal/` 🟢 **VERSIÓN FINAL DEL PROYECTO**
**Esta carpeta contiene la implementación final del sistema**, utilizada para ejecutar el experimento completo en un entorno controlado real. Integra todos los módulos anteriores en un único flujo automatizado:

- `frames/`: Capturas clave durante el vuelo
- `masks/`: Segmentaciones generadas en tiempo real
- `depth_maps/`: Estimaciones de profundidad por MiDaS
- `detections/`: Detección de partes y desperfectos
- `models_detections/`: Detecciones organizadas por modelo
- `detection_metrics.csv`: Métricas de las deteccioens de los modelos
- `metrics.csv`: Métricas del vuelo
- `informe_final.pdf`: Informe final automático en base a los datos capturados y procesados

> Este es el resultado consolidado del proyecto, empleado para evaluar el sistema en condiciones reales.

---

## 🔧 Tecnologías empleadas

- **Dron**: DJI Tello (controlado mediante `djitellopy`)
- **Modelos de IA**:
  - YOLOv8n-seg (segmentación semántica)
  - YOLOv8n (detección de partes/desperfectos)
  - MiDaS (estimación de profundidad monocular)
- **Frameworks**: PyTorch, OpenCV, PIL, Pandas, FPDF

---

## 📑 Objetivo del sistema

- Identificación automática del coche mediante segmentación
- Seguimiento del objeto con corrección de trayectoria y ajuste de distancia
- Aplicación localizada de modelos de detección de partes y daños
- Generación automatizada de un informe visual y cuantitativo

---

## 📷 Representación visual

- **Esquema general del sistema autónomo**  
  ![Diagrama del flujo del sistema](assets/esquema intro.png)

- **Capturas reales del vuelo autónomo**  
  ![Ejemplo de vuelo y seguimiento del objeto](assets/det_0028.png)

---

## 📌 Notas finales

Este repositorio representa un **proyecto académico de carácter experimental**, con un enfoque progresivo en la implementación de módulos individuales antes de su consolidación.  
Toda la experimentación ha sido realizada con hardware accesible y código modular, con el objetivo de facilitar futuras extensiones o aplicaciones industriales.

---

## 📫 Contacto

Desarrollado por María [Apellido],  
Trabajo de Fin de Grado en Ingeniería de Datos,  
Universidad [Nombre], 2025.
