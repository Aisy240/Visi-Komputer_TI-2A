import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# ✅ versi yang kompatibel
detector = HandDetector(mode=False, maxHands=1,
                        detectionCon=0.5, minTrackCon=0.5)

while True:
    ok, img = cap.read()
    if not ok:
        print("⚠️ Gagal membaca frame.")
        break

    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        count = sum(fingers)
        cv2.putText(img, f"Fingers: {count} {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hands + Fingers", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
