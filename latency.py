"""Ukur latensi: ResNet-18 vs MobileNetV3-Small (slide 22 pertemuan 3).

Cara pakai:
    python latency.py            # model terpilih = feature
    python latency.py partial    # atau pakai bobot mode lain

Yang diukur (batch 1, CPU, seperti 1 frame kamera robot):
  1. Model saja      : ResNet-18 (bobot hasil latihan) vs MobileNetV3-Small
  2. Alur lengkap    : undistort -> BGR ke RGB -> resize -> normalisasi -> ResNet-18
                       dibandingkan dengan anggaran 67 ms/frame (15 FPS, slide 29)
Hasil disimpan ke hasil/latensi.csv
"""

import csv
import glob
import json
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

FOLDER_HASIL = "hasil"
PEMANASAN = 10   # beberapa putaran awal selalu lebih lambat, tidak dihitung
ULANG = 100
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

mode = sys.argv[1] if len(sys.argv) > 1 else "feature"
kelas = json.load(open(os.path.join(FOLDER_HASIL, "kelas.json")))
n_kelas = len(kelas)


def ukur(fungsi):
    """Jalankan fungsi berkali-kali, kembalikan daftar waktu dalam milidetik."""
    for _ in range(PEMANASAN):
        fungsi()
    waktu = []
    for _ in range(ULANG):
        t0 = time.perf_counter()
        fungsi()
        waktu.append((time.perf_counter() - t0) * 1000)
    return waktu


def ringkas(nama, waktu):
    waktu = sorted(waktu)
    rata = statistics.mean(waktu)
    return {
        "yang_diukur": nama,
        "rata_ms": round(rata, 2),
        "median_ms": round(statistics.median(waktu), 2),
        "p95_ms": round(waktu[int(0.95 * len(waktu)) - 1], 2),
        "fps": round(1000 / rata, 1),
    }


def jumlah_parameter(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


# ---------- Model 1: ResNet-18 hasil latihan ----------
resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, n_kelas)
resnet.load_state_dict(torch.load(os.path.join(FOLDER_HASIL, f"resnet18_{mode}.pt")))
resnet.eval()

# ---------- Model 2: MobileNetV3-Small (pembanding) ----------
# Belum dilatih pada data kita; bobot tidak mempengaruhi kecepatan,
# jadi cukup arsitekturnya dengan head 2 kelas.
mobilenet = models.mobilenet_v3_small(weights=None)
mobilenet.classifier[-1] = nn.Linear(mobilenet.classifier[-1].in_features, n_kelas)
mobilenet.eval()

x = torch.randn(1, 3, 224, 224)
hasil = []
with torch.no_grad():
    hasil.append(ringkas("ResNet-18 (model saja)", ukur(lambda: resnet(x))))
    hasil.append(ringkas("MobileNetV3-Small (model saja)", ukur(lambda: mobilenet(x))))

# ---------- Alur lengkap seperti di robot ----------
data = np.load("calib.npz")
K, dist = data["K"], data["dist"]
lebar, tinggi = (int(v) for v in data["image_size"])
K_baru, _ = cv2.getOptimalNewCameraMatrix(K, dist, (lebar, tinggi), 0, (lebar, tinggi))
map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K_baru, (lebar, tinggi), cv2.CV_32FC1)

# Pakai satu foto val sebagai pengganti frame kamera 640x480
contoh = sorted(glob.glob(os.path.join("dataset_split", "val", "*", "*.png")))[0]
frame = cv2.imread(contoh)

tahap = {"undistort": [], "praproses (BGR->RGB, resize, normalisasi)": [], "inferensi ResNet-18": []}


def alur_lengkap():
    t0 = time.perf_counter()
    lurus = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    t1 = time.perf_counter()
    rgb = cv2.cvtColor(lurus, cv2.COLOR_BGR2RGB)          # OpenCV = BGR, model = RGB
    kecil = cv2.resize(rgb, (224, 224))
    arr = (kecil.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    t2 = time.perf_counter()
    with torch.no_grad():
        pred = resnet(tensor).argmax(dim=1).item()
    t3 = time.perf_counter()
    tahap["undistort"].append((t1 - t0) * 1000)
    tahap["praproses (BGR->RGB, resize, normalisasi)"].append((t2 - t1) * 1000)
    tahap["inferensi ResNet-18"].append((t3 - t2) * 1000)
    return pred


total = ukur(alur_lengkap)
for nama, w in tahap.items():
    hasil.append(ringkas(f"  tahap: {nama}", w[PEMANASAN:]))
hasil.append(ringkas("ALUR LENGKAP (undistort s.d. inferensi)", total))

# ---------- Tampilkan & simpan ----------
print(f"Perangkat: {platform.processor() or platform.machine()} | CPU threads: {torch.get_num_threads()}")
print(f"Parameter: ResNet-18 {jumlah_parameter(resnet):.1f} juta | "
      f"MobileNetV3-Small {jumlah_parameter(mobilenet):.1f} juta")
print(f"Bobot ResNet-18 yang dipakai: mode '{mode}'\n")
print(f"{'yang diukur':<52}{'rata(ms)':>9}{'median':>8}{'p95':>8}{'FPS':>7}")
for h in hasil:
    print(f"{h['yang_diukur']:<52}{h['rata_ms']:>9}{h['median_ms']:>8}{h['p95_ms']:>8}{h['fps']:>7}")

anggaran = 67
lengkap = hasil[-1]["rata_ms"]
status = "MASUK" if lengkap <= anggaran else "MELEBIHI"
print(f"\nAnggaran 15 FPS = {anggaran} ms/frame -> alur lengkap {lengkap} ms ({status} anggaran)")

with open(os.path.join(FOLDER_HASIL, "latensi.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["yang_diukur", "rata_ms", "median_ms", "p95_ms", "fps"])
    w.writeheader()
    w.writerows(hasil)
print(f"Tersimpan: {FOLDER_HASIL}/latensi.csv")
