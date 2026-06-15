"""
gui_app.py
==========
IFB 206 Komputasi Paralel | Evaluasi 3 | ITENAS Bandung

Antarmuka grafis PyQt5 untuk ParallelVision:
  - Pilih folder gambar
  - Atur operasi & parameter
  - Atur jumlah CPU core
  - Progress bar real-time
  - Preview hasil gambar (before/after)
  - Benchmark sequential vs paralel + grafik speedup
"""

import sys
import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSpinBox,
    QSlider, QProgressBar, QFileDialog, QGroupBox,
    QScrollArea, QFrame, QSplitter, QTabWidget, QTextEdit,
    QSizePolicy, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon

# Import engine kita
import sys
sys.path.insert(0, os.path.dirname(__file__))
from parallel_processor import ParallelImageProcessor, ProcessConfig, ImageResult

# ── Matplotlib untuk grafik benchmark ─────────────
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ═══════════════════════════════════════════════════
# WORKER THREAD (agar GUI tidak freeze)
# ═══════════════════════════════════════════════════

class ProcessWorker(QThread):
    progress    = pyqtSignal(int, int, object)   # current, total, result
    finished    = pyqtSignal(list, float)         # results, elapsed
    error       = pyqtSignal(str)

    def __init__(self, paths, config, n_workers, mode="parallel"):
        super().__init__()
        self.paths     = paths
        self.config    = config
        self.n_workers = n_workers
        self.mode      = mode

    def run(self):
        try:
            processor = ParallelImageProcessor(self.n_workers)

            def cb(cur, total, result):
                self.progress.emit(cur, total, result)

            if self.mode == "parallel":
                results, elapsed = processor.process(self.paths, self.config, cb)
            else:
                results, elapsed = processor.process_sequential(self.paths, self.config, cb)

            self.finished.emit(results, elapsed)
        except Exception as e:
            self.error.emit(str(e))


class BenchmarkWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)

    def __init__(self, paths, config):
        super().__init__()
        self.paths  = paths
        self.config = config

    def run(self):
        try:
            from multiprocessing import cpu_count
            max_w = cpu_count()
            processor = ParallelImageProcessor()
            cfg = self.config

            # Sequential
            self.status.emit("Menjalankan Sequential...")
            _, t_seq = processor.process_sequential(self.paths, cfg)

            # Paralel berbagai jumlah worker
            times_par = {}
            for w in range(1, max_w + 1):
                self.status.emit(f"Paralel {w} worker(s)...")
                processor.n_workers = w
                _, t = processor.process(self.paths, cfg)
                times_par[w] = t

            self.finished.emit({
                "t_seq":    t_seq,
                "t_par":    times_par,
                "max_w":    max_w,
                "n_images": len(self.paths),
            })
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════
# WIDGET: IMAGE PREVIEW
# ═══════════════════════════════════════════════════

def numpy_to_pixmap(img: np.ndarray, max_w=300, max_h=240) -> QPixmap:
    """Konversi numpy array BGR/Gray ke QPixmap."""
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = img.shape
    qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ImageCard(QFrame):
    """Kartu tampilan sebelum/sesudah satu gambar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #1e2433;
                border-radius: 8px;
                border: 1px solid #2d3452;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Row gambar
        img_row = QHBoxLayout()
        self.lbl_before = QLabel("Sebelum")
        self.lbl_after  = QLabel("Sesudah")
        for lbl in [self.lbl_before, self.lbl_after]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(160, 120)
            lbl.setStyleSheet("background:#111827; border-radius:4px; color:#6b7280;")
        img_row.addWidget(self.lbl_before)
        img_row.addWidget(self.lbl_after)
        layout.addLayout(img_row)

        # Info
        self.lbl_info = QLabel("")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color:#94a3b8; font-size:10px;")
        layout.addWidget(self.lbl_info)

    def update(self, img_path: str, result: ImageResult):
        before = cv2.imread(img_path)
        self.lbl_before.setPixmap(numpy_to_pixmap(before, 155, 115))
        if result.success and result.output_image is not None:
            self.lbl_after.setPixmap(numpy_to_pixmap(result.output_image, 155, 115))
        name = Path(img_path).name
        ms   = result.elapsed_ms
        status = "✓" if result.success else "✗"
        self.lbl_info.setText(f"{status} {name}  |  {result.operation.upper()}  |  {ms:.1f} ms")


# ═══════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2d3452;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #94a3b8;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #60a5fa; }
QPushButton {
    background: #1e40af;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover { background: #2563eb; }
QPushButton:disabled { background: #374151; color: #6b7280; }
QPushButton#danger { background: #7f1d1d; }
QPushButton#danger:hover { background: #991b1b; }
QPushButton#success { background: #065f46; }
QPushButton#success:hover { background: #047857; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #1e2433;
    border: 1px solid #2d3452;
    border-radius: 5px;
    padding: 5px 8px;
    color: #e2e8f0;
}
QComboBox::drop-down { border: none; }
QProgressBar {
    border: 1px solid #2d3452;
    border-radius: 5px;
    background: #1e2433;
    text-align: center;
    color: white;
}
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1d4ed8,stop:1 #7c3aed); border-radius: 4px; }
QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #60a5fa;
}
QLabel#subtitle { color: #64748b; font-size: 11px; }
QTextEdit {
    background: #0d1117;
    border: 1px solid #2d3452;
    border-radius: 6px;
    color: #4ade80;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}
QTabBar::tab {
    background: #1e2433;
    border: 1px solid #2d3452;
    border-bottom: none;
    padding: 6px 16px;
    border-radius: 4px 4px 0 0;
    color: #94a3b8;
}
QTabBar::tab:selected { background: #0f1117; color: #60a5fa; font-weight: bold; }
QScrollBar:vertical { background: #1e2433; width: 8px; }
QScrollBar::handle:vertical { background: #374151; border-radius: 4px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ParallelVision – Pemrosesan Citra Paralel | ITENAS IFB 206 | Azahra Dwi Putri Latifah | 152024159")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(DARK_STYLE)

        self.image_paths: list = []
        self.results:     list = []
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Header ──────────────────────────────
        header = QHBoxLayout()
        left_hdr = QVBoxLayout()
        lbl_title = QLabel("⚡ ParallelVision")
        lbl_title.setObjectName("title")
        lbl_sub = QLabel("Pemrosesan Citra Digital Paralel  ·  IFB 206 Komputasi Paralel  ·  ITENAS Bandung")
        lbl_sub.setObjectName("subtitle")
        left_hdr.addWidget(lbl_title)
        left_hdr.addWidget(lbl_sub)
        header.addLayout(left_hdr)
        header.addStretch()

        # CPU info
        from multiprocessing import cpu_count
        lbl_cpu = QLabel(f"CPU: {cpu_count()} core tersedia")
        lbl_cpu.setStyleSheet("color:#fbbf24; font-weight:bold; font-size:12px;")
        header.addWidget(lbl_cpu)
        root.addLayout(header)

        # ── Tabs ────────────────────────────────
        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._build_tab_process(), "🖼  Proses Gambar")
        tabs.addTab(self._build_tab_benchmark(), "📊  Benchmark")
        tabs.addTab(self._build_tab_log(), "📋  Log")

    # ─────────────────────────────────────────────
    # TAB 1: PROSES GAMBAR
    # ─────────────────────────────────────────────
    def _build_tab_process(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(10)

        # ── Panel kiri: Kontrol ─────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(280)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(10)

        # --- Pilih folder ---
        grp_folder = QGroupBox("📂 Input")
        fl = QVBoxLayout(grp_folder)
        self.lbl_folder = QLabel("Belum ada folder dipilih")
        self.lbl_folder.setWordWrap(True)
        self.lbl_folder.setStyleSheet("color:#64748b; font-size:11px;")
        btn_folder = QPushButton("Pilih Folder Gambar")
        btn_folder.clicked.connect(self._pick_folder)
        btn_sample = QPushButton("Gunakan Sample Images")
        btn_sample.setObjectName("success")
        btn_sample.clicked.connect(self._use_sample)
        fl.addWidget(self.lbl_folder)
        fl.addWidget(btn_folder)
        fl.addWidget(btn_sample)
        self.lbl_count = QLabel("0 gambar")
        self.lbl_count.setStyleSheet("color:#60a5fa; font-weight:bold;")
        fl.addWidget(self.lbl_count)
        ctrl_layout.addWidget(grp_folder)

        # --- Operasi ---
        grp_op = QGroupBox("⚙️ Operasi")
        ol = QGridLayout(grp_op)
        ol.addWidget(QLabel("Jenis:"), 0, 0)
        self.combo_op = QComboBox()
        self.combo_op.addItems([
            "canny", "dilasi", "erosi", "opening",
            "closing", "gradient", "blur", "otsu", "pipeline"
        ])
        ol.addWidget(self.combo_op, 0, 1)

        ol.addWidget(QLabel("Canny Low:"), 1, 0)
        self.spin_clow = QSpinBox(); self.spin_clow.setRange(0, 255); self.spin_clow.setValue(50)
        ol.addWidget(self.spin_clow, 1, 1)

        ol.addWidget(QLabel("Canny High:"), 2, 0)
        self.spin_chigh = QSpinBox(); self.spin_chigh.setRange(0, 255); self.spin_chigh.setValue(150)
        ol.addWidget(self.spin_chigh, 2, 1)

        ol.addWidget(QLabel("Morph Kernel:"), 3, 0)
        self.spin_kernel = QSpinBox(); self.spin_kernel.setRange(1, 31); self.spin_kernel.setSingleStep(2); self.spin_kernel.setValue(5)
        ol.addWidget(self.spin_kernel, 3, 1)

        ol.addWidget(QLabel("Iterasi:"), 4, 0)
        self.spin_iter = QSpinBox(); self.spin_iter.setRange(1, 10); self.spin_iter.setValue(1)
        ol.addWidget(self.spin_iter, 4, 1)

        ctrl_layout.addWidget(grp_op)

        # --- Paralel ---
        grp_par = QGroupBox("🔀 Komputasi Paralel")
        pl = QGridLayout(grp_par)
        from multiprocessing import cpu_count
        pl.addWidget(QLabel("Jumlah Worker:"), 0, 0)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, cpu_count())
        self.spin_workers.setValue(min(4, cpu_count()))
        pl.addWidget(self.spin_workers, 0, 1)
        self.lbl_workers_hint = QLabel(f"(max: {cpu_count()} core)")
        self.lbl_workers_hint.setStyleSheet("color:#64748b; font-size:10px;")
        pl.addWidget(self.lbl_workers_hint, 1, 0, 1, 2)
        ctrl_layout.addWidget(grp_par)

        # --- Output ---
        grp_out = QGroupBox("💾 Output")
        out_l = QVBoxLayout(grp_out)
        btn_out = QPushButton("Pilih Folder Output")
        btn_out.clicked.connect(self._pick_output)
        self.lbl_outdir = QLabel("output/")
        self.lbl_outdir.setStyleSheet("color:#64748b; font-size:10px;")
        out_l.addWidget(btn_out)
        out_l.addWidget(self.lbl_outdir)
        self.output_dir = "output"
        ctrl_layout.addWidget(grp_out)

        ctrl_layout.addStretch()

        # --- Tombol Jalankan ---
        self.btn_run = QPushButton("▶  Jalankan Paralel")
        self.btn_run.setFixedHeight(42)
        self.btn_run.setStyleSheet("font-size:14px; background:#7c3aed;")
        self.btn_run.clicked.connect(self._run_parallel)
        ctrl_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl_layout.addWidget(self.btn_stop)

        # Progress
        self.progress = QProgressBar()
        self.progress.setFixedHeight(22)
        ctrl_layout.addWidget(self.progress)

        self.lbl_status = QLabel("Siap")
        self.lbl_status.setStyleSheet("color:#94a3b8; font-size:11px;")
        ctrl_layout.addWidget(self.lbl_status)

        layout.addWidget(ctrl)

        # ── Panel kanan: Preview hasil ──────────
        right = QWidget()
        right_layout = QVBoxLayout(right)

        lbl_preview = QLabel("Preview Hasil (Before → After)")
        lbl_preview.setStyleSheet("font-weight:bold; color:#60a5fa;")
        right_layout.addWidget(lbl_preview)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_widget = QWidget()
        self.preview_grid   = QGridLayout(self.preview_widget)
        self.preview_grid.setSpacing(8)
        scroll.setWidget(self.preview_widget)
        right_layout.addWidget(scroll)

        # Stat bar
        self.lbl_stat = QLabel("")
        self.lbl_stat.setStyleSheet("color:#fbbf24; font-weight:bold; padding:4px;")
        right_layout.addWidget(self.lbl_stat)

        layout.addWidget(right)
        self.cards = []
        return tab

    # ─────────────────────────────────────────────
    # TAB 2: BENCHMARK
    # ─────────────────────────────────────────────
    def _build_tab_benchmark(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.btn_bench = QPushButton("🚀  Jalankan Benchmark (Sequential vs Paralel)")
        self.btn_bench.setFixedHeight(38)
        self.btn_bench.clicked.connect(self._run_benchmark)
        top.addWidget(self.btn_bench)
        self.lbl_bench_status = QLabel("Pilih gambar dulu, lalu jalankan benchmark.")
        self.lbl_bench_status.setStyleSheet("color:#94a3b8;")
        top.addWidget(self.lbl_bench_status)
        layout.addLayout(top)

        if HAS_MPL:
            self.fig_bench  = Figure(figsize=(10, 4), facecolor="#0f1117")
            self.canvas_bench = FigureCanvas(self.fig_bench)
            layout.addWidget(self.canvas_bench)
        else:
            layout.addWidget(QLabel("matplotlib tidak tersedia – install: pip install matplotlib"))

        self.txt_bench = QTextEdit()
        self.txt_bench.setReadOnly(True)
        self.txt_bench.setFixedHeight(120)
        layout.addWidget(self.txt_bench)
        return tab

    # ─────────────────────────────────────────────
    # TAB 3: LOG
    # ─────────────────────────────────────────────
    def _build_tab_log(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        btn_clear = QPushButton("Hapus Log")
        btn_clear.setObjectName("danger")
        btn_clear.clicked.connect(self.log_text.clear)
        layout.addWidget(btn_clear)
        return tab

    # ─────────────────────────────────────────────
    # SLOT: Pilih folder, sample, dll
    # ─────────────────────────────────────────────
    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder Gambar")
        if folder:
            self._load_images_from_folder(folder)

    def _use_sample(self):
        base = os.path.dirname(os.path.dirname(__file__))
        sample_dir = os.path.join(base, "sample_images")
        if os.path.isdir(sample_dir):
            self._load_images_from_folder(sample_dir)
        else:
            self._log("❌ Folder sample_images tidak ditemukan")

    def _load_images_from_folder(self, folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self.image_paths = [
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in exts
        ]
        self.image_paths.sort()
        self.lbl_folder.setText(folder)
        self.lbl_count.setText(f"{len(self.image_paths)} gambar ditemukan")
        self._log(f"📂 Folder: {folder} | {len(self.image_paths)} gambar")

    def _pick_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder Output")
        if folder:
            self.output_dir = folder
            self.lbl_outdir.setText(folder)

    def _make_config(self) -> ProcessConfig:
        return ProcessConfig(
            operation      = self.combo_op.currentText(),
            canny_low      = self.spin_clow.value(),
            canny_high     = self.spin_chigh.value(),
            morph_kernel   = self.spin_kernel.value(),
            morph_iterations = self.spin_iter.value(),
            output_dir     = self.output_dir,
            save_result    = True,
        )

    # ─────────────────────────────────────────────
    # SLOT: Jalankan Paralel
    # ─────────────────────────────────────────────
    def _run_parallel(self):
        if not self.image_paths:
            self.lbl_status.setText("⚠ Pilih folder gambar dulu!")
            return
        self._clear_preview()
        self.progress.setMaximum(len(self.image_paths))
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Memproses...")
        self._log(f"▶ Mulai: {len(self.image_paths)} gambar | {self.spin_workers.value()} worker | op={self.combo_op.currentText()}")

        config = self._make_config()
        self.worker = ProcessWorker(self.image_paths, config, self.spin_workers.value())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.lbl_status.setText("Dihentikan")
            self._log("■ Proses dihentikan")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_progress(self, cur, total, result: ImageResult):
        self.progress.setValue(cur)
        self.lbl_status.setText(f"Memproses {cur}/{total}: {Path(result.path).name}")
        self._log(f"  [{cur}/{total}] {Path(result.path).name} → {result.elapsed_ms:.1f} ms {'✓' if result.success else '✗'}")
        self._add_preview_card(result)

    def _on_finished(self, results, elapsed):
        self.results = results
        ok    = sum(1 for r in results if r.success)
        total_ms = sum(r.elapsed_ms for r in results)
        self.lbl_stat.setText(
            f"✅ Selesai  |  {ok}/{len(results)} berhasil  |  "
            f"Total: {elapsed:.2f}s  |  Rata-rata: {total_ms/max(1,len(results)):.1f}ms/gambar  |  "
            f"Throughput: {len(results)/elapsed:.1f} gambar/detik"
        )
        self._log(f"✅ Selesai dalam {elapsed:.2f}s | {ok}/{len(results)} berhasil | Output: {self.output_dir}")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText(f"Selesai – {elapsed:.2f}s")

    def _on_error(self, msg):
        self._log(f"❌ Error: {msg}")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ─────────────────────────────────────────────
    # PREVIEW CARDS
    # ─────────────────────────────────────────────
    def _clear_preview(self):
        for card in self.cards:
            card.deleteLater()
        self.cards.clear()

    def _add_preview_card(self, result: ImageResult):
        card = ImageCard()
        card.update(result.path, result)
        row = len(self.cards) // 2
        col = len(self.cards) % 2
        self.preview_grid.addWidget(card, row, col)
        self.cards.append(card)

    # ─────────────────────────────────────────────
    # SLOT: Benchmark
    # ─────────────────────────────────────────────
    def _run_benchmark(self):
        if not self.image_paths:
            self.lbl_bench_status.setText("⚠ Pilih gambar dulu!")
            return
        self.btn_bench.setEnabled(False)
        self.lbl_bench_status.setText("Menjalankan benchmark, harap tunggu...")
        self._log("📊 Benchmark dimulai...")

        config = self._make_config()
        self.bench_worker = BenchmarkWorker(self.image_paths, config)
        self.bench_worker.finished.connect(self._on_bench_finished)
        self.bench_worker.error.connect(lambda e: self._log(f"❌ {e}"))
        self.bench_worker.status.connect(lambda s: self.lbl_bench_status.setText(s))
        self.bench_worker.start()

    def _on_bench_finished(self, data):
        self.btn_bench.setEnabled(True)
        t_seq  = data["t_seq"]
        t_par  = data["t_par"]
        max_w  = data["max_w"]
        n_img  = data["n_images"]

        workers = sorted(t_par.keys())
        times   = [t_par[w] for w in workers]
        speedups = [t_seq / t for t in times]

        summary = (
            f"Sequential: {t_seq:.3f}s\n"
            + "\n".join(f"  {w} worker: {t_par[w]:.3f}s  (speedup {t_seq/t_par[w]:.2f}×)" for w in workers)
            + f"\nBest speedup: {max(speedups):.2f}× dengan {workers[speedups.index(max(speedups))]} worker"
        )
        self.txt_bench.setText(summary)
        self._log("📊 Benchmark selesai\n" + summary)
        self.lbl_bench_status.setText(f"Selesai! Best speedup: {max(speedups):.2f}×")

        if HAS_MPL:
            self.fig_bench.clear()
            ax1 = self.fig_bench.add_subplot(121)
            ax2 = self.fig_bench.add_subplot(122)
            for ax in [ax1, ax2]:
                ax.set_facecolor("#161b22")
                ax.tick_params(colors="#94a3b8", labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#2d3452")

            # Grafik waktu
            ax1.axhline(t_seq, color="#f87171", ls="--", lw=1.5, label=f"Sequential ({t_seq:.2f}s)")
            ax1.bar(workers, times, color="#60a5fa", alpha=0.85, width=0.6)
            ax1.set_title("Waktu Eksekusi (s)", color="#e2e8f0", fontsize=10)
            ax1.set_xlabel("Jumlah Worker", color="#94a3b8", fontsize=8)
            ax1.legend(fontsize=8, facecolor="#1e2433", labelcolor="white")

            # Grafik speedup
            ax2.plot(workers, speedups, 'o-', color="#4ade80", lw=2, ms=7)
            ax2.plot(workers, workers, '--', color="#fbbf24", lw=1, alpha=0.6, label="Ideal Linear")
            ax2.set_title("Speedup vs Jumlah Worker", color="#e2e8f0", fontsize=10)
            ax2.set_xlabel("Jumlah Worker", color="#94a3b8", fontsize=8)
            ax2.legend(fontsize=8, facecolor="#1e2433", labelcolor="white")

            self.fig_bench.patch.set_facecolor("#0f1117")
            self.fig_bench.tight_layout()
            self.canvas_bench.draw()

    # ─────────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────────
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")


# ═══════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════

def main():
    # Wajib untuk multiprocessing di Windows
    import multiprocessing
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
