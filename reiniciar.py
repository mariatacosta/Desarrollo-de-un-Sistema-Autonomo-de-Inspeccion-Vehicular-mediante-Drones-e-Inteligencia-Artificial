import time
import csv
import cv2
from djitellopy import Tello

#NO FUNCIONA LEL

dron = Tello()
dron.connect()

dron.end()  # Apagar la conexión con el dron
time.sleep(2)
dron.connect()  # Vuelve a conectar el dron
