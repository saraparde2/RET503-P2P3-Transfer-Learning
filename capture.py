"""Ambil foto dataset yang sudah di-undistort + catat ke metadata.csv.

Cara pakai:
    python capture.py <kelas> <kondisi_cahaya> [lokasi]
Contoh:
    python capture.py botol terang
    python capture.py tissue redup kamar

SPASI = simpan foto, Q = keluar.
"""

import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np

NOMOR_KAMERA = 1
KELAS_VALID = ["botol", "kertas", "tissue"]
CAHAYA_VALID = ["terang", "redup", "jendela", "bayangan"]
FOLDER_DATASET = "dataset_raw"
FILE_METADATA = os.path.join(FOLDER_DATASET, "metadata.csv")

# ---------- 1. Baca argumen dari terminal ----------
if len(sys.argv) < 3:
    print("Cara pakai: python capture.py <kelas> <kondisi_cahaya> [lokasi]")
    print("Contoh   : python capture.py botol terang")
    sys.exit(1)

kelas = sys.argv[1].lower()
cahaya = sys.argv[2].lower()
lokasi = sys.argv[3].lower() if len(sys.argv) > 3 else "rumah"

if kelas not in KELAS_VALID:
    print(f"Kelas '{kelas}' tidak dikenal. Pilih salah satu: {KELAS_VALID}")
    sys.exit(1)
if cahaya not in CAHAYA_VALID:
    print(f"Kondisi cahaya '{cahaya}' tidak dikenal. Pilih salah satu: {CAHAYA_VALID}")
    sys.exit(1)
if len(sys.argv) > 4 or not lokasi.isalnum():
    print("Perintah terlalu panjang atau lokasi aneh. Cek lagi baris perintahnya.")
    sys.exit(1)

# ---------- 2. Siapkan undistort dari calib.npz ----------
data = np.load("calib.npz")
K, dist = data["K"], data["dist"]
lebar, tinggi = (int(v) for v in data["image_size"])
K_baru, _ = cv2.getOptimalNewCameraMatrix(K, dist, (lebar, tinggi), 0, (lebar, tinggi))
map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K_baru, (lebar, tinggi), cv2.CV_32FC1)

# ---------- 3. Siapkan folder dan metadata.csv ----------
folder_kelas = os.path.join(FOLDER_DATASET, kelas)
os.makedirs(folder_kelas, exist_ok=True)

if not os.path.exists(FILE_METADATA):
    with open(FILE_METADATA, "w", newline="") as f:
        csv.writer(f).writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])

# Nomor urut melanjutkan jumlah foto yang sudah ada di folder kelas ini
nomor = len(os.listdir(folder_kelas)) + 1

# ---------- 4. Loop kamera ----------
cap = cv2.VideoCapture(NOMOR_KAMERA)
print(f"Kelas: {kelas} | Cahaya: {cahaya} | Lokasi: {lokasi}")
print("SPASI = simpan, Q = keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Gagal membaca kamera. Cek NOMOR_KAMERA.")
        break
    if (frame.shape[1], frame.shape[0]) != (lebar, tinggi):
        print("Resolusi kamera berbeda dengan saat kalibrasi:", frame.shape)
        break

    lurus = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    # Teks info hanya di tampilan, BUKAN di foto yang disimpan
    tampilan = lurus.copy()
    info = f"{kelas} | {cahaya} | total kelas ini: {nomor - 1}"
    cv2.putText(tampilan, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Capture dataset - SPASI simpan, Q keluar", tampilan)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        tanggal = datetime.now().strftime("%Y%m%d")
        nama_file = f"{kelas}_{tanggal}_{lokasi}_{cahaya}_{nomor:03d}.png"
        cv2.imwrite(os.path.join(folder_kelas, nama_file), lurus)
        with open(FILE_METADATA, "a", newline="") as f:
            csv.writer(f).writerow([nama_file, kelas, tanggal, cahaya])
        print("Tersimpan:", nama_file)
        nomor += 1

cap.release()
cv2.destroyAllWindows()
