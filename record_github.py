from time import sleep, time
from djitellopy import Tello
import datetime
import os
import cv2
import csv

# Directorio donde se guardarán las imágenes y el CSV
save_dir = os.path.expanduser("~/Desktop/tello_data")
os.makedirs(save_dir, exist_ok=True)

# Archivo CSV
csv_path = os.path.join(save_dir, f'tello_{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}.csv')

def save_metrics(drone, csv_writer):
    """ Guarda métricas del dron en el CSV """
    try:
        state = drone.get_current_state()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([timestamp] + [state[key] for key in sorted(state.keys())])
    except Exception as e:
        print(f"Error al obtener métricas: {e}")


def capture_image(drone, image_counter):
    """ Captura y guarda una imagen de la cámara del dron """
    try:

        frame_reader = drone.get_frame_read()
        sleep(3)  # Esperar un poco para recibir frames
        if frame_reader.frame is None or frame_reader.frame.size == 0:
            print("Advertencia: No hay frames disponibles todavía.")
        frame = frame_reader.frame


        if frame is None or frame.size == 0:
            print("Advertencia: No se recibió un fotograma válido")
            return

        image_path = os.path.join(save_dir, f'image_{image_counter}.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Imagen guardada: {image_path}")
    except Exception as e:
        print(f"Error al capturar imagen: {e}")


def test():
    drone = Tello()
    image_counter = 0

    try:
        drone.connect()
        print(f"Nivel de batería: {drone.get_battery()}%")

        drone.streamon()  # Activar la cámara
        sleep(5)  # Esperar a que el flujo de video se estabilice

        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            # Escribir encabezado con todas las métricas disponibles
            state_keys = sorted(drone.get_current_state().keys())
            csv_writer.writerow(["timestamp"] + state_keys)

            drone.takeoff()
            start_time = time()
            interval = 5  # Captura cada 5 segundos

            while time() - start_time < 40:  # Ejecutar durante 20 segundos como prueba
                save_metrics(drone, csv_writer)
                capture_image(drone, image_counter)
                image_counter += 1
                sleep(interval)

            drone.land()

    except Exception as ex:
        print(f"Error: {ex}")

    finally:
        drone.streamoff()
        drone.end()
        print(f"Datos guardados en {csv_path}")

if __name__ == '__main__':
    test()
