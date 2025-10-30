import cv2
import mediapipe as mp
import time

# Inisialisasi modul Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Buka kamera (0 = kamera utama)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Kamera tidak bisa dibuka. Coba ganti index ke 1 atau 2.")

# Inisialisasi Pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Gagal membaca frame dari kamera.")
            break

        # Konversi BGR ke RGB (wajib untuk Mediapipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Gambar landmark jika terdeteksi
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        # Hitung FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Tampilkan FPS di layar
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Tampilkan hasil
        cv2.imshow("Pose Detection", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Tutup semua jendela
cap.release()
cv2.destroyAllWindows()