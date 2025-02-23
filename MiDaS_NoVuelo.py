from djitellopy import Tello
import cv2
import time
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Crear carpetas para guardar imágenes
os.makedirs("capturas_originales", exist_ok=True)
os.makedirs("profundidades_midas", exist_ok=True)

# Inicializar el dron
drone = Tello()
drone.connect()
print(f"🔋 Batería del dron: {drone.get_battery()}%")

# Iniciar el stream de video
drone.streamoff()
drone.streamon()
time.sleep(2)

# El dron NO despega
print("🚁 Dron en el suelo, capturando datos sin despegar...")

# Cargar modelo MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Transformación de imagen para MiDaS
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    input_batch = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

# Contadores para guardar imágenes
image_count = 0
start_time = time.time()

try:
    while True:
        # Obtener el frame del dron
        frame = drone.get_frame_read().frame
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        depth_map = estimate_depth(frame)

        # Normalizar profundidad para visualizarla mejor
        depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_visual = np.uint8(depth_visual)
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

        # Mostrar imágenes
        cv2.imshow("Imagen Capturada", frame)
        cv2.imshow("Mapa de Profundidad (MiDaS)", depth_colormap)

        # Guardar imágenes cada 2 segundos
        if time.time() - start_time > 2:
            original_path = f"capturas_originales/frame_{image_count}.jpg"
            depth_path = f"profundidades_midas/depth_{image_count}.jpg"
            cv2.imwrite(original_path, frame)
            cv2.imwrite(depth_path, depth_colormap)
            print(f"📸 Imágenes guardadas: {original_path}, {depth_path}")
            image_count += 1
            start_time = time.time()

        # Si se presiona 'q', finalizar captura
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Finalizando captura de datos...")
            break

except KeyboardInterrupt:
    print("🔴 Interrupción detectada, finalizando captura...")

# Finalizar
cv2.destroyAllWindows()
drone.streamoff()
