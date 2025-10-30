import cv2, numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

detector = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)

blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3
EYE_AR_THRESHOLD = 0.32  # Lebih sensitif
is_closed = False
ear_values = []
MAX_EAR_HISTORY = 5

while True:
    ok, img = cap.read()
    if not ok:
        print("⚠️ Frame gagal dibaca.")
        break

    img, faces = detector.findFaceMesh(img, draw=True)
    if faces:
        face = faces[0]
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)

        # Smoothing
        ear_values.append(ear)
        if len(ear_values) > MAX_EAR_HISTORY:
            ear_values.pop(0)
        ear_smooth = sum(ear_values) / len(ear_values)

        cv2.putText(img, f"EAR(L): {ear_smooth:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if ear_smooth < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        cv2.putText(img, f"Blink: {blink_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("FaceMesh + EAR", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
