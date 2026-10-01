import os

import cv2
from datetime import datetime

NOMOR_KAMERA = 1              # 0 = webcam USB (ganti kalau perlu)
FOLDER = "folder_papan"        # foto disimpan di folder ini

os.makedirs(FOLDER, exist_ok=True)
cap = cv2.VideoCapture(NOMOR_KAMERA)
jumlah = len(os.listdir(FOLDER))

print("Tekan SPASI untuk memotret, tekan Q untuk berhenti.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Gagal baca kamera. Coba ganti NOMOR_KAMERA.")
        break

    tampilan = frame.copy()
    cv2.putText(tampilan, f"Foto tersimpan: {jumlah}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Kamera - SPASI = foto, Q = keluar", tampilan)

    tombol = cv2.waitKey(1)
    if tombol == ord(" "):
        nama = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        cv2.imwrite(os.path.join(FOLDER, nama), frame)
        jumlah += 1
        print("Tersimpan:", nama)
    elif tombol == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
