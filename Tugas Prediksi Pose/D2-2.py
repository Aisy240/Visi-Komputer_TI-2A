import cv2
import math
from cvzone.PoseModule import PoseDetector

# Gunakan kamera index 1 (karena hasil cek kamu menunjukkan index 1 aktif)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 2 jika tetap hitam.")

detector = PoseDetector(mode=False, smooth=True,
                        detectionCon=0.5, trackCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Gagal membaca frame.")
        break

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    if lmList:
        center = bboxInfo["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # --- JARAK MANUAL ---
        x1, y1 = lmList[11][1], lmList[11][2]  # bahu kiri
        x2, y2 = lmList[15][1], lmList[15][2]  # pergelangan tangan kiri
        length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img, f"Length: {length}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # --- SUDUT MANUAL ---
        x1, y1 = lmList[11][1], lmList[11][2]
        x2, y2 = lmList[13][1], lmList[13][2]
        x3, y3 = lmList[15][1], lmList[15][2]

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"Angle: {int(angle)}°", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
