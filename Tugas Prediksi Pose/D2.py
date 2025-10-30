import cv2
from cvzone.PoseModule import PoseDetector

# Buka kamera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi pose detector
detector = PoseDetector(mode=False, smooth=True,
                        detectionCon=0.5, trackCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Gagal membaca frame.")
        break

    # Deteksi pose
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    if lmList:
        center = bboxInfo["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Hitung jarak antar landmark
        length, img, info = detector.findDistance(11, 15, img)

        # Hitung sudut (bahu–siku–pergelangan tangan)
        angle, img = detector.findAngle(11, 13, 15, img)

        # Tampilkan hasil di layar
        cv2.putText(img, f"Angle: {int(angle)}°", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Length: {int(length)}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
