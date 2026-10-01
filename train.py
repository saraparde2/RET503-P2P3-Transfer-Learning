"""Latih ResNet-18 dengan 3 mode transfer learning (slide 23 pertemuan 3).

| Mode    | Bobot awal | Yang dilatih | Learning rate |
| feature | ImageNet   | fc saja      | 1e-3          |
| partial | ImageNet   | layer4 + fc  | 1e-4 / 1e-3   |
| scratch | acak       | semua        | 1e-3          |

Cara pakai:
    python train.py            # jalankan ketiga mode berurutan
    python train.py feature    # hanya satu mode (feature / partial / scratch)

Semua hasil disimpan di folder hasil/:
    log_<mode>.csv        akurasi & loss per epoch
    ringkasan.csv         tabel hasil 3 mode (untuk README)
    grafik_akurasi.png    grafik akurasi per epoch
    salah_<mode>.csv      foto val yang salah ditebak (untuk analisis)
    resnet18_<mode>.pt    bobot model terbaik
    kelas.json            urutan nama kelas
"""

import csv
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

FOLDER_DATA = "dataset_split"
FOLDER_HASIL = "hasil"
EPOCHS = 10
BATCH = 16
SEED = 42
SEMUA_MODE = ["feature", "partial", "scratch"]

# Normalisasi harus sama dengan data ImageNet tempat ResNet-18 dilatih (slide 13)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Augmentasi hanya untuk train (slide 23): RandomResizedCrop, HorizontalFlip, ColorJitter.
# scale=(0.6, 1.0) supaya potongan acak tidak membuang objek yang ada di tepi foto.
TF_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# Val (dan nanti di robot): cukup resize -> normalisasi, tanpa acak.
# Catatan: datasets.ImageFolder membaca foto sebagai RGB. Di robot, frame OpenCV
# (BGR) harus diubah dulu ke RGB sebelum masuk model.
TF_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def set_seed(seed):
    """Supaya hasil bisa diulang dengan angka yang (hampir) sama."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def buat_model(mode, n_kelas):
    if mode == "scratch":
        m = models.resnet18(weights=None)  # bobot acak
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():           # bekukan seluruh backbone
            p.requires_grad = False
        if mode == "partial":
            for p in m.layer4.parameters():  # buka blok terakhir
                p.requires_grad = True
    # Head baru sesuai jumlah kelas kita; otomatis bisa dilatih
    m.fc = nn.Linear(m.fc.in_features, n_kelas)
    return m


def buat_optimizer(mode, m):
    # Hanya masukkan parameter yang memang dilatih (slide 11)
    if mode == "feature":
        return torch.optim.Adam(m.fc.parameters(), lr=1e-3)
    if mode == "partial":
        return torch.optim.Adam([
            {"params": m.layer4.parameters(), "lr": 1e-4},
            {"params": m.fc.parameters(), "lr": 1e-3},
        ])
    return torch.optim.Adam(m.parameters(), lr=1e-3)


def mode_latih(m, mode):
    """Lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah (slide 11)."""
    if mode == "feature":
        m.eval()
        m.fc.train()
    elif mode == "partial":
        m.eval()
        m.layer4.train()
        m.fc.train()
    else:
        m.train()


def evaluasi(m, loader):
    m.eval()
    benar, total, tebakan = 0, 0, []
    with torch.no_grad():
        for x, y in loader:
            pred = m(x).argmax(dim=1)
            benar += (pred == y).sum().item()
            total += y.size(0)
            tebakan.extend(pred.tolist())
    return benar / total, tebakan


def latih_satu_mode(mode, ds_train, ds_val):
    set_seed(SEED)
    kelas = ds_train.classes
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=0)

    m = buat_model(mode, len(kelas))
    opt = buat_optimizer(mode, m)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    log = []
    terbaik, epoch_terbaik, epoch_90 = -1.0, 0, None
    path_model = os.path.join(FOLDER_HASIL, f"resnet18_{mode}.pt")
    mulai = time.perf_counter()

    print(f"\n=== Mode: {mode} ===")
    for epoch in range(1, EPOCHS + 1):
        mode_latih(m, mode)
        total_loss, benar, total = 0.0, 0, 0
        for x, y in dl_train:
            opt.zero_grad()
            out = m(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            benar += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        sched.step()

        acc_train = benar / total
        acc_val, _ = evaluasi(m, dl_val)
        waktu = time.perf_counter() - mulai
        log.append([epoch, round(total_loss / total, 4), round(acc_train, 4), round(acc_val, 4), round(waktu, 1)])
        print(f"epoch {epoch:2d} | loss {total_loss / total:.4f} | "
              f"acc train {acc_train:.3f} | acc val {acc_val:.3f} | {waktu:.0f} dtk")

        if acc_val > terbaik:
            terbaik, epoch_terbaik = acc_val, epoch
            torch.save(m.state_dict(), path_model)
        if epoch_90 is None and acc_val >= 0.9:
            epoch_90 = epoch

    waktu_total = time.perf_counter() - mulai

    with open(os.path.join(FOLDER_HASIL, f"log_{mode}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "waktu_kumulatif_detik"])
        w.writerows(log)

    # Catat foto val yang salah ditebak oleh model terbaik (bahan analisis README)
    m.load_state_dict(torch.load(path_model))
    _, tebakan = evaluasi(m, dl_val)
    with open(os.path.join(FOLDER_HASIL, f"salah_{mode}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nama_file", "label_asli", "tebakan"])
        for (path, y), p in zip(ds_val.samples, tebakan):
            if p != y:
                w.writerow([os.path.basename(path), kelas[y], kelas[p]])

    return {
        "mode": mode,
        "akurasi_val_terbaik": round(terbaik, 4),
        "epoch_terbaik": epoch_terbaik,
        "epoch_pertama_val_90": epoch_90 if epoch_90 is not None else "-",
        "waktu_latih_detik": round(waktu_total, 1),
    }


def simpan_ringkasan(hasil_baru):
    path = os.path.join(FOLDER_HASIL, "ringkasan.csv")
    semua = {}
    if os.path.exists(path):  # gabungkan dengan hasil mode yang sudah pernah dijalankan
        with open(path, newline="") as f:
            for baris in csv.DictReader(f):
                semua[baris["mode"]] = baris
    for h in hasil_baru:
        semua[h["mode"]] = h
    kolom = ["mode", "akurasi_val_terbaik", "epoch_terbaik", "epoch_pertama_val_90", "waktu_latih_detik"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        for mode in SEMUA_MODE:
            if mode in semua:
                w.writerow({k: semua[mode][k] for k in kolom})

    print("\n=== Ringkasan ===")
    print(f"{'mode':<9}{'val terbaik':>12}{'epoch':>7}{'epoch>=90%':>12}{'waktu(dtk)':>12}")
    for mode in SEMUA_MODE:
        if mode in semua:
            r = semua[mode]
            print(f"{mode:<9}{float(r['akurasi_val_terbaik']):>12.3f}{str(r['epoch_terbaik']):>7}"
                  f"{str(r['epoch_pertama_val_90']):>12}{float(r['waktu_latih_detik']):>12.1f}")


def buat_grafik():
    plt.figure(figsize=(8, 5))
    warna = {"feature": "tab:blue", "partial": "tab:green", "scratch": "tab:red"}
    for mode in SEMUA_MODE:
        path = os.path.join(FOLDER_HASIL, f"log_{mode}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        ep = [int(d["epoch"]) for d in data]
        plt.plot(ep, [float(d["val_acc"]) for d in data], "-o", color=warna[mode], label=f"{mode} (val)")
        plt.plot(ep, [float(d["train_acc"]) for d in data], "--", color=warna[mode], alpha=0.5, label=f"{mode} (train)")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.ylim(0, 1.05)
    plt.title("Akurasi per epoch - ResNet-18, 3 mode")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_HASIL, "grafik_akurasi.png"), dpi=150)
    print(f"Grafik disimpan: {FOLDER_HASIL}/grafik_akurasi.png")


if __name__ == "__main__":
    mode_dipilih = sys.argv[1:] or SEMUA_MODE
    for mode in mode_dipilih:
        if mode not in SEMUA_MODE:
            print(f"Mode '{mode}' tidak dikenal. Pilih: {SEMUA_MODE}")
            sys.exit(1)

    os.makedirs(FOLDER_HASIL, exist_ok=True)
    ds_train = datasets.ImageFolder(os.path.join(FOLDER_DATA, "train"), transform=TF_TRAIN)
    ds_val = datasets.ImageFolder(os.path.join(FOLDER_DATA, "val"), transform=TF_VAL)
    print(f"Kelas: {ds_train.classes} | train: {len(ds_train)} foto | val: {len(ds_val)} foto")

    with open(os.path.join(FOLDER_HASIL, "kelas.json"), "w") as f:
        json.dump(ds_train.classes, f)

    hasil = [latih_satu_mode(mode, ds_train, ds_val) for mode in mode_dipilih]
    simpan_ringkasan(hasil)
    buat_grafik()
