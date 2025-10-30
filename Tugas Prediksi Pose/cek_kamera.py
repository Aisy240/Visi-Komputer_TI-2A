import cv2

print("🔍 Mendeteksi kamera yang tersedia...\n")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✅ Kamera {i} TERSEDIA")
        cap.release()
    else:
        print(f"❌ Kamera {i} tidak bisa dibuka")
