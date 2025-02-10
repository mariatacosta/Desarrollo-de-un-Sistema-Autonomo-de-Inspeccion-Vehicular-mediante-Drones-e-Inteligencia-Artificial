import time
import csv
import cv2
from djitellopy import Tello

# Inicializar dron
dron = Tello()
dron.connect()
time.sleep(2)

dron.send_command_with_return("command")
time.sleep(2)
print("Batería:", dron.get_battery(), "%")

dron.streamoff()
dron.streamon()
time.sleep(3)

# Configuración de grabación de video
video_filename = "vuelo.avi"
video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"XVID"), 20, (640, 480))

# Inicializar archivo CSV
with open("telemetria.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "acc_x", "acc_y", "acc_z", "pos_x", "pos_y", "altura", "bateria"])


def guardar_datos(timestamp, dron):
    with open("telemetria.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            dron.get_acceleration_x(), dron.get_acceleration_y(), dron.get_acceleration_z(),
            0, 0,  # Posiciones ficticias (se pueden actualizar según la necesidad)
            dron.get_height(),
            dron.get_battery()
        ])


def capturar_imagen():
    frame = dron.get_frame_read().frame
    if frame is None:
        return None

    timestamp = time.time()
    filename = f"imagen_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return timestamp


# Iniciar vuelo
dron.takeoff()
time.sleep(5)  # Esperar estabilización

# Iniciar captura de video
start_time = time.time()
ultimo_guardado = start_time

while True:
    try:
        frame = dron.get_frame_read().frame
        if frame is None:
            print("Reiniciando stream de video...")
            dron.streamoff()
            time.sleep(2)
            dron.streamon()
            time.sleep(3)
            continue

        frame = cv2.resize(frame, (640, 480))
        video_writer.write(frame)
        cv2.imshow("Imagen", frame)
    except Exception as e:
        print(f"Error al procesar el frame: {e}")
        continue

    # Captura de imagen y datos cada 5 segundos
    if time.time() - ultimo_guardado > 5:
        timestamp = capturar_imagen()
        if timestamp:
            guardar_datos(timestamp, dron)
        ultimo_guardado = time.time()

    # Modificar recorrido
    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        dron.move_left(35)
        time.sleep(2)
    if elapsed_time > 15:
        dron.send_control_command("cw 90")
        time.sleep(2)
    if elapsed_time > 20:
        break

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Aterrizaje seguro
for _ in range(3):
    try:
        dron.land()
        break
    except:
        print("Reintentando aterrizar...")
        time.sleep(2)

dron.streamoff()
dron.end()
video_writer.release()
cv2.destroyAllWindows()
