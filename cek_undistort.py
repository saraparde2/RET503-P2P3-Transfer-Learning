"""Cek hasil kalibrasi: tampilkan webcam ASLI (kiri) dan HASIL UNDISTORT (kanan)."""

import cv2
import numpy as np

NOMOR_KAMERA = 1  # sama dengan yang dipakai saat memotret papan

# 1. Ambil hasil kalibrasi
data = np.load("calib.npz")
K = data["K"]
dist = data["dist"]
lebar, tinggi = data["image_size"]  # ukuran saat kalibrasi (640, 480)

# 2. Siapkan "peta pelurusan" sekali saja di awal (lebih cepat untuk video).
#    alpha=0 -> bagian hitam di pinggir hasil undistort dipotong,
#    jadi gambar hasil tetap penuh tanpa tepi hitam.
K_baru, _ = cv2.getOptimalNewCameraMatrix(K, dist, (lebar, tinggi), 0, (lebar, tinggi))
map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K_baru, (lebar, tinggi), cv2.CV_32FC1)

cap = cv2.VideoCapture(NOMOR_KAMERA)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Gagal membaca kamera")
        break

    # Resolusi harus sama dengan saat kalibrasi
    if (frame.shape[1], frame.shape[0]) != (lebar, tinggi):
        print("Resolusi kamera berbeda dengan saat kalibrasi:", frame.shape)
        break

    lurus = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    gabung = np.hstack([frame, lurus])  # kiri: asli, kanan: undistort
    cv2.putText(gabung, "ASLI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(gabung, "UNDISTORT", (lebar + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Cek undistort - tekan Q untuk keluar", gabung)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
