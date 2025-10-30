import cv2, numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

MODE = "squat"  # tekan 'm' untuk toggle ke "pushup"
KNEE_DOWN, KNEE_UP = 80, 160
DOWN_R, UP_R = 0.85, 1.00
SAMPLE_OK = 4

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Pastikan DroidCam aktif!")

detector = PoseDetector()
count, state = 0, "up"
debounce = deque(maxlen=6)

def ratio_pushup(lm):
    sh = np.array(lm[11][1:3])
    wr = np.array(lm[15][1:3])
    hp = np.array(lm[23][1:3])
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)

while True:
    ok, img = cap.read()
    if not ok:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)
    flag = None

    if lmList:
        if MODE == "squat":
            # ✅ versi terbaru: (p1, p2, p3, img)
            angL, img = detector.findAngle(23, 25, 27, img, draw=True)
            angR, img = detector.findAngle(24, 26, 28, img, draw=True)

            ang = (angL + angR) / 2.0
            if ang < KNEE_DOWN: flag = "down"
            elif ang > KNEE_UP: flag = "up"
            cv2.putText(img, f"Knee: {ang:5.1f}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            r = ratio_pushup(lmList)
            if r < DOWN_R: flag = "down"
            elif r > UP_R: flag = "up"
            cv2.putText(img, f"Ratio: {r:4.2f}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        debounce.append(flag)
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"; count += 1

    cv2.putText(img, f"Mode: {MODE.upper()} Count: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(img, f"State: {state}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Pose Counter", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

cap.release()
cv2.destroyAllWindows()
