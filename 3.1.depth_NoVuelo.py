import cv2

ip = '0.0.0.0'  # Escuchar en todas las interfaces
port = 11111    # Puerto por defecto Tello

url = f"udp://{ip}:{port}"

print(f"Intentando abrir stream en {url}")

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("No se pudo abrir el stream")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se recibió frame")
        break
    cv2.imshow("Stream Tello", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
