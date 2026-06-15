# ⚡ ParallelVision – Pemrosesan Citra Digital Paralel

> **IFB 206 Komputasi Paralel | Evaluasi 3 | Semester Genap 2025/2026**
> Institut Teknologi Nasional (ITENAS) Bandung
> Dosen: Lisa Kristiana, Ph.D.

---

## 👥 Tim
| Azahra Dwi Putri Latifah | 152024159 |

---

## 📌 Deskripsi Proyek

**ParallelVision** adalah aplikasi pemrosesan citra digital yang menggunakan **komputasi paralel** (`multiprocessing` Python) untuk memproses sekumpulan gambar secara bersamaan di beberapa CPU core secara simultan.

Aplikasi ini menerapkan filter **Canny Edge Detection** dan berbagai **operasi morfologi** (dilasi, erosi, opening, closing, gradient morfologi), serta dilengkapi antarmuka grafis **PyQt5** dan mode CLI interaktif.

---

## 🧠 Konsep Komputasi Paralel

```
100 Gambar Input
      │
      ▼
┌──────────────────────────────────────────────────┐
│            multiprocessing.Pool                  │
│                                                  │
│  Worker 1  │ Worker 2  │ Worker 3  │  Worker 4  │
│  img 1–25  │ img 26–50 │ img 51–75 │ img 76–100 │
│  [Canny]   │ [Canny]   │ [Canny]   │  [Canny]   │
└──────────────────────────────────────────────────┘
      │
      ▼
100 Gambar Output (~3–5× lebih cepat dari sequential)
```

### Mengapa Paralel Lebih Cepat?

Setiap gambar diproses **secara independen** — tidak ada data yang perlu dibagi antar gambar. Kondisi ini disebut **embarrassingly parallel**, sehingga beban kerja bisa dibagi sempurna ke semua CPU core yang tersedia.

Berdasarkan **Hukum Amdahl**: jika 90% pekerjaan dapat diparalelkan, dengan 4 core → speedup teoritis ≈ **3.1× lebih cepat**.

---

## ⚙️ Operasi yang Didukung

| Operasi | Keterangan |
|---------|------------|
| `canny` | Deteksi tepi dengan dua threshold (low & high) |
| `dilasi` | Memperluas area foreground (piksel putih) |
| `erosi` | Menyusutkan area foreground |
| `opening` | Erosi → Dilasi: menghilangkan noise kecil |
| `closing` | Dilasi → Erosi: menutup lubang kecil |
| `gradient` | Tepi objek = dilasi − erosi |
| `blur` | Gaussian blur untuk pelembutan gambar |
| `otsu` | Segmentasi otomatis dengan threshold Otsu |
| `pipeline` | Gabungan: Canny + Dilasi + Closing |

---

## 🏗️ Arsitektur Sistem

```
ParallelVision/
├── src/
│   ├── parallel_processor.py   ← Engine inti multiprocessing
│   └── gui_app.py              ← Antarmuka grafis PyQt5
├── tests/
│   └── test_processor.py       ← 25 unit test (semua passed ✅)
├── sample_images/              ← 20 gambar contoh siap pakai
├── output/                     ← Hasil pemrosesan (dibuat otomatis)
├── docs/
│   └── VIDEO_GUIDE.md          ← Panduan video demonstrasi
├── run_cli.py                  ← Entry point CLI berwarna
├── requirements.txt
└── README.md
```

---

## 🚀 Instalasi & Cara Penggunaan

### Prasyarat
- Python 3.9 atau lebih baru
- pip

### 1. Clone Repository
```bash
git clone https://github.com/Student-Embedded-Control-and-AI-Fest/CuffnCode.git
cd CuffnCode
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Mode CLI

```bash
# Proses 20 gambar secara paralel dengan Canny Edge Detection
python run_cli.py --folder sample_images --op canny --workers 4

# Operasi morfologi: pipeline lengkap
python run_cli.py --folder sample_images --op pipeline --workers 4

# Erosi
python run_cli.py --folder sample_images --op erosi --workers 4

# Lihat semua opsi
python run_cli.py --help
```

### 4. Jalankan Benchmark (Sequential vs Paralel)
```bash
python run_cli.py --folder sample_images --benchmark
```

Contoh output:
```
Sequential    : 2.341s
1 worker      : 2.089s  Speedup 1.12×  ▓▓▓▓▓▓
2 worker      : 1.203s  Speedup 1.95×  ▓▓▓▓▓▓▓▓▓▓
4 worker      : 0.698s  Speedup 3.35×  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

🏆 Best: 4 worker → speedup 3.35× (efisiensi 84%)
```

### 5. Jalankan GUI PyQt5
```bash
python src/gui_app.py
```

Fitur GUI:
- Pilih folder gambar input
- Atur operasi & parameter (threshold, kernel size, iterasi)
- Atur jumlah CPU worker
- Progress bar real-time
- Preview before/after setiap gambar
- Tab benchmark dengan grafik speedup otomatis

### 6. Jalankan Unit Test
```bash
python tests/test_processor.py
```
Output: `Ran 25 tests in 0.720s → OK`

---

## 📊 Hasil Benchmark (8 Core CPU)

| Mode | Waktu | Speedup |
|------|-------|---------|
| Sequential | 2.341s | 1.00× |
| 1 Worker | 2.089s | 1.12× |
| 2 Worker | 1.203s | 1.95× |
| 4 Worker | 0.698s | **3.35×** |
| 8 Worker | 0.431s | **5.43×** |

> Diuji dengan 20 gambar 256×256px, operasi Canny, CPU 8 core.

---

## 🔬 Penerapan Komputasi Paralel

Proyek ini menerapkan komputasi paralel melalui:

1. **`multiprocessing.Pool`** — membagi daftar gambar ke beberapa proses Python yang berjalan di core berbeda secara bersamaan (bukan threading, sehingga tidak terkena GIL Python).

2. **`Pool.imap_unordered`** — gambar yang selesai lebih cepat langsung dikembalikan tanpa menunggu gambar lain, memaksimalkan throughput.

3. **Worker function terpisah** — setiap proses menjalankan `_process_single_image()` secara independen tanpa shared state, sehingga tidak perlu lock/mutex.

4. **Progress callback real-time** — hasil tiap worker diteruskan ke main process via `imap_unordered` untuk ditampilkan di progress bar GUI/CLI.

---

## 📚 Referensi

- Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE TPAMI, 8(6), 679–698.
- Haralick & Shapiro (1992). *Computer and Robot Vision* – Bab Morfologi Matematika.
- Python `multiprocessing` Documentation: https://docs.python.org/3/library/multiprocessing.html
- OpenCV Documentation: https://docs.opencv.org/4.x/
- Amdahl, G. (1967). *Validity of the single processor approach to achieving large scale computing capabilities*.