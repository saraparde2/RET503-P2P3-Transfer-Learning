"""Menjalankan kalibrasi pada folder_papan lalu menyimpan hasilnya ke calib.npz."""

import numpy as np
from calibration_utils import calibrate_images

# display=False: tidak menampilkan foto satu per satu, langsung hitung.
hasil = calibrate_images("folder_papan/*.jpg", display=False)

print(f"Foto: {len(hasil.image_paths)}, terdeteksi: {hasil.detected_images}")
print(f"RMS: {hasil.rms_error:.3f} piksel")
print("Camera matrix (K):\n", hasil.camera_matrix)
print("Distortion coefficients:\n", hasil.distortion_coefficients)

# Simpan hanya yang dibutuhkan untuk undistort nanti.
np.savez(
    "calib.npz",
    K=hasil.camera_matrix,
    dist=hasil.distortion_coefficients,
    image_size=np.array(hasil.image_size),  # (lebar, tinggi) saat kalibrasi
    rms=hasil.rms_error,
)
print("Tersimpan ke calib.npz")
