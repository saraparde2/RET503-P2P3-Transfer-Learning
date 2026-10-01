"""Bagi dataset_raw menjadi train dan val, dengan cara yang mengurangi data leakage.

Kenapa tidak diacak biasa?
Foto dengan nomor berurutan biasanya diambil beruntun dengan posisi mirip.
Kalau diacak, foto "kembaran" bisa masuk ke train DAN val sekaligus, sehingga
akurasi val terlihat terlalu bagus (slide 24 pertemuan 3).

Cara di sini: untuk setiap kelas dan setiap kondisi cahaya, foto diurutkan
berdasarkan nomor, lalu BLOK TERAKHIR (sekitar 20%) dijadikan val.
Jadi val berisi "sesi" pemotretan yang tidak pernah dilihat model saat latihan,
dan tiap kondisi cahaya tetap terwakili di train maupun val.

Hasil:
    dataset_split/train/<kelas>/...
    dataset_split/val/<kelas>/...
    dataset_split/split.csv   (daftar file -> train/val)
"""

import csv
import os
import shutil
from collections import defaultdict

FOLDER_RAW = "dataset_raw"
FOLDER_SPLIT = "dataset_split"
PORSI_VAL = 0.2

# Mulai dari folder kosong supaya hasil selalu sama dan tidak ada sisa lama
if os.path.exists(FOLDER_SPLIT):
    shutil.rmtree(FOLDER_SPLIT)

# Kelas = nama subfolder di dataset_raw (otomatis ikut kalau nanti ada kelas baru)
kelas_list = sorted(
    d for d in os.listdir(FOLDER_RAW) if os.path.isdir(os.path.join(FOLDER_RAW, d))
)

baris_csv = []
ringkasan = defaultdict(lambda: {"train": 0, "val": 0})

for kelas in kelas_list:
    folder = os.path.join(FOLDER_RAW, kelas)
    # Kelompokkan per kondisi cahaya. Format nama: kelas_tanggal_lokasi_cahaya_NNN.png
    per_kondisi = defaultdict(list)
    for nama in os.listdir(folder):
        if not nama.lower().endswith(".png"):
            continue
        bagian = os.path.splitext(nama)[0].split("_")
        kondisi, nomor = bagian[-2], int(bagian[-1])
        per_kondisi[kondisi].append((nomor, nama))

    for kondisi, daftar in sorted(per_kondisi.items()):
        daftar.sort()  # urut berdasarkan nomor = urutan pemotretan
        n_val = max(1, round(len(daftar) * PORSI_VAL))
        for i, (_, nama) in enumerate(daftar):
            bagian_set = "val" if i >= len(daftar) - n_val else "train"
            tujuan = os.path.join(FOLDER_SPLIT, bagian_set, kelas)
            os.makedirs(tujuan, exist_ok=True)
            shutil.copy2(os.path.join(folder, nama), os.path.join(tujuan, nama))
            baris_csv.append([nama, kelas, kondisi, bagian_set])
            ringkasan[kelas][bagian_set] += 1

with open(os.path.join(FOLDER_SPLIT, "split.csv"), "w", newline="") as f:
    tulis = csv.writer(f)
    tulis.writerow(["nama_file", "kelas", "kondisi_cahaya", "set"])
    tulis.writerows(baris_csv)

print(f"{'kelas':<10}{'train':>7}{'val':>6}")
for kelas in kelas_list:
    print(f"{kelas:<10}{ringkasan[kelas]['train']:>7}{ringkasan[kelas]['val']:>6}")
print(f"Selesai. Hasil di folder '{FOLDER_SPLIT}'.")
