# Dokumen Desain Awal · PBL Trash Picking Robot

**Tim:** PBL Trash Picking Robot · **Disusun oleh:** Sara Devina Pardede (4222411053) · **Versi:** 1 (3 Oktober 2026, akan terus diperbarui)

> Nilai bertanda * adalah rencana/estimasi awal dan akan diperbarui setelah pengukuran pada robot.

## 1. Misi projek

Robot beroda dengan lengan dan *gripper* yang mencari sampah di lantai, mendekatinya, menjepit, lalu memasukkannya ke keranjang di atas robot dan membuangnya di titik pembuangan (alur: mendekat → posisi → buka → jepit → angkat → masukkan → buang).

**Peran persepsi.** Kamera menjadi satu-satunya sensor untuk menemukan sampah. Sistem persepsi harus (1) mengenali jenis sampah dan (2) memberi posisinya di citra, supaya sistem kemudi bisa mengarahkan robot mendekat dan lengan bisa diposisikan tepat di atas objek. Tahap saat ini (P3) adalah klasifikasi; tahap berikutnya adalah deteksi objek dengan YOLO agar posisi objek (*bounding box*) tersedia.

## 2. Kelas objek

| Kelas | Contoh foto | Status data |
|---|---|---|
| botol plastik | `dataset_raw/botol/botol_20260930_rumah_terang_010.png` | 66 citra |
| tissue | `dataset_raw/tissue/tissue_20260930_rumah_terang_008.png` | 60 citra |
| kertas | _belum diambil_ | direncanakan |

## 3. Kamera & dudukan

| Item | Nilai |
|---|---|
| Kamera | Webcam USB (pengganti karena kamera khusus tidak tersedia) |
| Resolusi | 640 × 480, dikalibrasi (RMS 0,817 px; `calib.npz`) |
| Posisi dudukan | Bagian depan bodi robot, di depan pangkal lengan, menghadap ke depan-bawah* |
| Tinggi dari lantai | ± 25 cm* |
| Sudut ke bawah | ± 35–40° dari horizontal* |
| Jarak kerja | ± 20–60 cm di depan robot (jangkauan lengan dan area mendekat)* |

## 4. Unit komputasi

| Item | Nilai |
|---|---|
| Perangkat pengembangan | Laptop, CPU AMD 4 thread, tanpa GPU |
| Perangkat onboard robot | Laptop yang sama dipasang/dibawa di robot, terhubung ke webcam via USB dan ke mikrokontroler motor/servo via USB serial* |
| Mode daya | Baterai laptop (komputasi) terpisah dari baterai robot (motor dan servo); Windows mode *Best performance* saat beroperasi* |
| Memori | 8 GB RAM* |
| Aktuator | Penggerak: 1 motor dengan sistem kemudi (*steering*), 4 roda dengan diferensial. Lengan: 3 sumbu + rotasi dasar (bisa berputar ke belakang menuju keranjang) + *gripper* |

## 5. Target kinerja

| Metrik | Target |
|---|---|
| Akurasi klasifikasi (val) | ≥ 90% |
| mAP@0,5 deteksi (setelah YOLO) | ≥ 0,7* |
| FPS onboard (alur lengkap) | ≥ 15 FPS (≤ 67 ms per frame) |
| Latensi ROS 2 kamera → perintah aktuator | ≤ 150 ms, jika integrasi memakai ROS 2* |

## 6. Kandidat model

| Model | Alasan | Hasil awal |
|---|---|---|
| ResNet-18 (klasifikasi) | Bobot ImageNet tersedia, akurat dengan data sedikit | Akurasi val 100% (mode *feature*), alur lengkap 74,2 ms (13,5 FPS) di CPU, sedikit di atas target |
| MobileNetV3-Small (klasifikasi) | ±4,4× lebih cepat, cocok untuk komputer robot tanpa GPU | 15,9 ms per inferensi; akurasi belum diuji |
| YOLO versi kecil/nano (deteksi) | Memberi kelas **dan** posisi objek sekaligus, yang dibutuhkan untuk mengarahkan robot dan lengan | Direncanakan setelah anotasi data (P4) |

## 7. Strategi transfer learning

Mulai dengan **feature extraction** (backbone ImageNet dibekukan, hanya lapisan akhir dilatih). Alasannya: data per kelas masih sedikit (± 60 citra), latihan paling cepat, risiko *overfitting* paling rendah, dan pada eksperimen P3 sudah mencapai akurasi val 100%. Jika kelas yang mirip (kertas vs tissue) membuat akurasi turun, naik ke **partial fine-tuning** (`layer4` + head dilatih). Melatih dari nol tidak dipakai karena pada data yang sama hanya mencapai 91,7% setelah 10 epoch.

## 8. Rencana data

- Sekarang: botol 66, tissue 60 citra, ≥ 50 per kelas, dengan `metadata.csv` (nama_file, kelas, tanggal, kondisi_cahaya).
- Target berikutnya: ≥ 100 citra per kelas termasuk **kertas**.
- Variasi: beberapa objek berbeda per kelas, posisi tengah dan tepi frame, jarak dekat–jauh, cahaya terang/redup/bayangan, latar keramik, lantai, meja, dan objek pengganggu.
- Untuk YOLO: anotasi *bounding box* (format YOLO/COCO) dan pengambilan data di **area kerja robot yang sebenarnya**, dari dudukan kamera robot.

## 9. Risiko dan mitigasi

| Risiko | Mitigasi |
|---|---|
| Kertas dan tissue mirip (putih, tipis) sehingga sering tertukar | Tambah data kertas dengan ciri khas (lipatan, tulisan, kertas berwarna); gunakan partial fine-tuning; analisis citra yang salah tebak |
| Latensi melebihi 67 ms di komputer tanpa GPU | Pakai MobileNetV3-Small atau YOLO nano; kecilkan resolusi input; proses tidak setiap frame |
| Webcam pengganti: *auto exposure*/fokus berubah dan dudukan bisa bergeser | Dudukan kamera dikunci mati; kalibrasi ulang bila posisi/resolusi berubah; kunci exposure bila webcam mendukung |
| Model belajar dari latar atau lokasi, bukan dari objeknya | Samakan campuran latar di semua kelas; ambil data di lokasi operasi robot; uji dengan objek dan lokasi baru |
| Botol bening sulit terlihat di lantai terang | Perbanyak contoh botol bening; atur sudut kamera agar pantulan dan kontur botol terlihat |
