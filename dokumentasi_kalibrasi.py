"""Buat folder dokumentasi untuk tugas kalibrasi kamera (pertemuan 2).

Cara pakai (di folder yang sama dengan calib.npz dan folder_papan):
    python dokumentasi_kalibrasi.py

Hasil:
    dokumentasi_kalibrasi/
    ├── deteksi_papan/     semua foto papan yang sudutnya terdeteksi (titik & garis warna)
    ├── distorted/         contoh foto ASLI dari webcam (belum dikoreksi)
    ├── undistorted/       foto yang sama setelah dikoreksi (undistort)
    ├── perbandingan/      kiri asli, kanan undistort, dalam satu gambar
    └── hasil_kalibrasi.txt   angka K, distorsi, RMS, daftar foto terdeteksi/gagal
"""

import glob
import os
import shutil

import cv2
import numpy as np

from calibration_utils import CHECKERBOARD  # (7, 7), sama dengan saat kalibrasi

FOLDER_FOTO = "folder_papan"
FOLDER_DOK = "dokumentasi_kalibrasi"
JUMLAH_CONTOH = 6  # berapa foto untuk contoh distorted/undistorted

# Mulai dari folder kosong
if os.path.exists(FOLDER_DOK):
    shutil.rmtree(FOLDER_DOK)
for sub in ["deteksi_papan", "distorted", "undistorted", "perbandingan"]:
    os.makedirs(os.path.join(FOLDER_DOK, sub))

# ---------- Hasil kalibrasi yang dipakai di seluruh tugas ----------
data = np.load("calib.npz")
K, dist = data["K"], data["dist"]
lebar, tinggi = (int(v) for v in data["image_size"])
rms = float(data["rms"])

# alpha=1: semua piksel asli dipertahankan, jadi tepi hitam melengkung terlihat.
# Ini sengaja, supaya efek koreksinya jelas di dokumentasi (seperti slide 30, cara 1).
K_baru, _ = cv2.getOptimalNewCameraMatrix(K, dist, (lebar, tinggi), 1, (lebar, tinggi))

# ---------- 1. Foto deteksi sudut papan catur ----------
semua_foto = sorted(glob.glob(os.path.join(FOLDER_FOTO, "*.jpg")))
terdeteksi, gagal = [], []
for path in semua_foto:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    nama = os.path.basename(path)
    if found:
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, found)
        cv2.imwrite(os.path.join(FOLDER_DOK, "deteksi_papan", nama), img)
        terdeteksi.append(path)
    else:
        gagal.append(nama)

# ---------- 2. Contoh distorted vs undistorted ----------
# Ambil beberapa foto yang tersebar merata dari daftar yang terdeteksi
if terdeteksi:
    idx = np.linspace(0, len(terdeteksi) - 1, min(JUMLAH_CONTOH, len(terdeteksi))).astype(int)
    for i in idx:
        path = terdeteksi[i]
        nama = os.path.splitext(os.path.basename(path))[0]
        asli = cv2.imread(path)
        lurus = cv2.undistort(asli, K, dist, None, K_baru)

        cv2.imwrite(os.path.join(FOLDER_DOK, "distorted", f"{nama}_distorted.jpg"), asli)
        cv2.imwrite(os.path.join(FOLDER_DOK, "undistorted", f"{nama}_undistorted.jpg"), lurus)

        gabung = np.hstack([asli.copy(), lurus.copy()])
        cv2.putText(gabung, "DISTORTED (asli)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(gabung, "UNDISTORTED", (lebar + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(FOLDER_DOK, "perbandingan", f"{nama}_perbandingan.jpg"), gabung)

# ---------- 3. Ringkasan angka ----------
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
k1, k2, p1, p2, k3 = dist.ravel()[:5]
with open(os.path.join(FOLDER_DOK, "hasil_kalibrasi.txt"), "w") as f:
    f.write("HASIL KALIBRASI KAMERA (webcam USB)\n")
    f.write("===================================\n")
    f.write(f"Resolusi              : {lebar} x {tinggi}\n")
    f.write(f"Pola papan (sudut dalam): {CHECKERBOARD[0]} x {CHECKERBOARD[1]} (papan {CHECKERBOARD[0] + 1}x{CHECKERBOARD[1] + 1} kotak)\n")
    f.write(f"Foto dipakai          : {len(semua_foto)}, terdeteksi: {len(terdeteksi)}, gagal: {len(gagal)}\n")
    f.write(f"RMS reprojection error: {rms:.3f} piksel\n\n")
    f.write("Matriks kamera K:\n")
    f.write(np.array2string(K, precision=3, suppress_small=True) + "\n")
    f.write(f"  fx = {fx:.2f}, fy = {fy:.2f}, cx = {cx:.2f}, cy = {cy:.2f}\n\n")
    f.write("Koefisien distorsi [k1, k2, p1, p2, k3]:\n")
    f.write(f"  {k1:.5f}, {k2:.5f}, {p1:.5f}, {p2:.5f}, {k3:.5f}  (k3 dikunci = 0)\n\n")
    f.write("Catatan metode:\n")
    f.write("  - Deteksi sudut: cv2.findChessboardCornersSB (EXHAUSTIVE | ACCURACY)\n")
    f.write("  - Kalibrasi: cv2.calibrateCamera dengan flag CALIB_FIX_K3\n")
    f.write("  - Foto undistorted di folder ini memakai alpha=1 (tepi hitam terlihat)\n\n")
    f.write("Foto yang gagal dideteksi:\n")
    for nama in gagal:
        f.write(f"  {nama}\n")

print(f"Terdeteksi {len(terdeteksi)} dari {len(semua_foto)} foto")
print(f"Dokumentasi tersimpan di folder '{FOLDER_DOK}'")
