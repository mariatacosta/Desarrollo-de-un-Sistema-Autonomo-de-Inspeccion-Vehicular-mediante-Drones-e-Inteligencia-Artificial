import time
import csv
import cv2
from djitellopy import Tello

# # Configuración
# ALTURA_VUELO = 40  # cm
# LADO_LARGO = 50  # cm (con margen)
# LADO_CORTO = 40  # cm (con margen)
# TIEMPO_CAPTURA = 5  # segundos

# Inicializar dron
dron = Tello()
dron.connect()
time.sleep(4)

# Inicia el modo SDK (si no está activado)
dron.send_command_with_return("command")
time.sleep(4)

print("Batería:", dron.get_battery(), "%")

#capturamos datos
dron.streamoff()
dron.streamon()
time.sleep(2)

while True:
    img = dron.get_frame_read().frame
    img = cv2.resize(img, (640, 480))
    cv2.imshow("Imagen", img)
    # waitKey: delay between every frame
    #si presionamos q se para el programa
    if cv2.waitKey(5) & 0xFF == ord('q'):
        dron.streamoff()
        break

def guardar_datos(timestamp, acc, gyro, pos_x, pos_y):
    """Guarda los datos en un archivo CSV"""
    with open("telemetria.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, acc, gyro, pos_x, pos_y])

def capturar_imagen():
    """Captura y guarda una imagen con timestamp"""
    frame = dron.get_frame_read().frame
    timestamp = time.time()
    filename = f"imagen_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return timestamp

# Iniciar vuelo
dron.takeoff()
time.sleep(3)

#"Mitad" lado corto 1
dron.move_left(35)
time.sleep(1)
# dron.streamon()

dron.send_control_command("cw 90")
time.sleep(2)

# #Lado largo 1
# dron.move_left(100)
# time.sleep(1)
# dron.send_control_command("cw 90")
# time.sleep(2)
#
# #Lado corto 2
# dron.move_left(50)
# time.sleep(1)
# dron.send_control_command("cw 90")
# time.sleep(3)
#
# #Lado largo 2
# dron.move_left(100)
# time.sleep(1)
# dron.send_control_command("cw 90")
# time.sleep(2)
# #
# #"Mitad" lado corto 1 (Volver a Inicio)
# dron.move_left(35)
# time.sleep(1)

#Aterrizar
dron.land()
dron.streamoff()
dron.end()
