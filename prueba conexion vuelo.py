import time
from djitellopy import Tello

dron = Tello()
dron.connect()
time.sleep(2)

print("Intentando despegar...")
dron.takeoff()
time.sleep(2)

print("Intentando aterrizar...")
dron.land()
dron.end()
