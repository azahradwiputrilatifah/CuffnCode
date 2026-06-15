# 🎬 Panduan Video Demo – ParallelVision
## IFB 206 Komputasi Paralel | Evaluasi 3

---

## Durasi Target: 25–30 detik

---

## 🎞️ Storyboard

### [0–4 dtk] Opening – Judul
**Rekam**: Ketik di terminal, biarkan banner muncul
```bash
python3 run_cli.py --help
```
**Teks overlay**: "ParallelVision – IFB 206 Komputasi Paralel | ITENAS"

---

### [4–12 dtk] Demo CLI Paralel – INI BAGIAN PALING PENTING
**Rekam perintah ini persis** (progress bar akan terlihat keren di video):
```bash
python3 run_cli.py --folder sample_images --op canny --workers 4
```
Tampilkan:
- Progress bar bergerak `[████████░░░░░░] 10/20 ✓`
- Setiap gambar selesai dengan waktu ms
- Output akhir: "✅ Selesai: 20/20 | Total: 0.4s | Throughput: 46 gambar/detik"

---

### [12–20 dtk] Demo Benchmark – HIGHLIGHT PARALEL
**Rekam perintah benchmark**:
```bash
python3 run_cli.py --folder sample_images --op pipeline --benchmark
```
Tampilkan tabel speedup yang muncul:
```
Sequential    : 2.341s
1 worker      : 2.089s  Speedup 1.12×  ▓▓▓▓▓▓
2 worker      : 1.203s  Speedup 1.95×  ▓▓▓▓▓▓▓▓▓▓
4 worker      : 0.698s  Speedup 3.35×  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
🏆 Best: 4 worker → speedup 3.35×
```

---

### [20–25 dtk] Tampilkan GUI (opsional, jika bisa buka display)
```bash
python3 src/gui_app.py
```
Atau: tampilkan screenshot GUI yang sudah diambil sebelumnya.
Tunjukkan:
- Tab "Proses Gambar" dengan before/after
- Tab "Benchmark" dengan grafik speedup

---

### [25–30 dtk] Closing
**Rekam**: Unit test berjalan
```bash
python3 tests/test_processor.py -v
```
Tampilkan: "Ran 25 tests in 0.720s → OK"

**Teks overlay**:
- "github.com/Student-Embedded-Control-and-AI-Fest/CuffnCode"
- "ITENAS Bandung 2025/2026"

---

## 📱 Caption Instagram

```
⚡ ParallelVision – Digital Image Processing
Evaluasi 3 | IFB 206 Komputasi Paralel | ITENAS Bandung

Aplikasi pemrosesan citra paralel berbasis Python:
🔹 Canny Edge Detection + Operasi Morfologi
🔹 multiprocessing.Pool → bagi beban ke semua CPU core
🔹 Speedup hingga 3–5× lebih cepat dari sequential
🔹 GUI PyQt5 + CLI dengan progress bar real-time
🔹 25 unit test ✅

#KomputasiParalel #Python #OpenCV #ImageProcessing
#ITENAS #Informatika #CannyEdge #Multiprocessing
#ComputerVision #Bandung
```

---

## 🎬 Tips Rekaman

- Perbesar font terminal ke 14–16pt sebelum rekam (lebih mudah dibaca)
- Windows: `Win + G` untuk record, atau OBS Studio
- Linux: `SimpleScreenRecorder` atau `OBS`
- Mac: `Cmd + Shift + 5`
- Edit dengan CapCut → tambah musik lo-fi ringan
