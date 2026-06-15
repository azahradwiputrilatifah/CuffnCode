"""
parallel_processor.py
=====================
IFB 206 Komputasi Paralel | Evaluasi 3 | ITENAS Bandung
Tema: Pemrosesan Citra Digital Paralel

Engine inti yang membagi pekerjaan pemrosesan citra ke beberapa
CPU core menggunakan multiprocessing.Pool.

Operasi yang didukung:
  - Canny Edge Detection
  - Morfologi: Dilasi, Erosi, Opening, Closing, Gradient
  - Gaussian Blur
  - Threshold (Otsu)
  - Kombinasi pipeline (Edge + Morph)
"""

import cv2
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════

@dataclass
class ProcessConfig:
    """Konfigurasi pipeline pemrosesan."""
    operation: str        = "canny"      # canny | dilasi | erosi | opening | closing | gradient | blur | otsu | pipeline
    canny_low: int        = 50
    canny_high: int       = 150
    morph_kernel: int     = 5            # ukuran kernel morfologi (px)
    morph_iterations: int = 1
    blur_ksize: int       = 5            # harus ganjil
    output_dir: str       = "output"
    save_result: bool     = True
    colormap: int         = cv2.COLORMAP_INFERNO   # untuk visualisasi hasil


@dataclass
class ImageResult:
    """Hasil pemrosesan satu gambar."""
    path: str
    success: bool
    operation: str
    elapsed_ms: float
    input_shape: Tuple = field(default_factory=tuple)
    output_image: Optional[np.ndarray] = field(default=None, repr=False)
    error: str = ""


# ═══════════════════════════════════════════════════
# WORKER FUNCTION (dijalankan di tiap proses)
# ═══════════════════════════════════════════════════

def _process_single_image(args: Tuple) -> ImageResult:
    """
    Worker function – dipanggil oleh setiap proses child.
    Tidak boleh menggunakan objek yang tidak picklable (mis. lambda, Qt).

    args = (image_path, config_dict, worker_id)
    """
    image_path, cfg_dict, worker_id = args
    cfg = ProcessConfig(**cfg_dict)

    t0 = time.perf_counter()

    try:
        # ── Baca gambar ──────────────────────────
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Tidak bisa membaca: {image_path}")

        shape = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Pilih operasi ─────────────────────────
        op = cfg.operation.lower()
        result_gray = _apply_operation(gray, op, cfg)

        # ── Colormap untuk output berwarna ────────
        if len(result_gray.shape) == 2:
            result_color = cv2.applyColorMap(result_gray, cfg.colormap)
        else:
            result_color = result_gray

        # ── Simpan hasil ──────────────────────────
        if cfg.save_result:
            os.makedirs(cfg.output_dir, exist_ok=True)
            fname = Path(image_path).stem + f"_{op}.jpg"
            out_path = os.path.join(cfg.output_dir, fname)
            cv2.imwrite(out_path, result_color)

        elapsed = (time.perf_counter() - t0) * 1000

        return ImageResult(
            path=image_path,
            success=True,
            operation=op,
            elapsed_ms=elapsed,
            input_shape=shape,
            output_image=result_color,
        )

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return ImageResult(
            path=image_path,
            success=False,
            operation=cfg.operation,
            elapsed_ms=elapsed,
            error=str(e),
        )


def _apply_operation(gray: np.ndarray, op: str, cfg: ProcessConfig) -> np.ndarray:
    """Terapkan satu operasi pemrosesan ke gambar grayscale."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.morph_kernel, cfg.morph_kernel)
    )

    if op == "canny":
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        return cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    elif op == "dilasi":
        return cv2.dilate(gray, kernel, iterations=cfg.morph_iterations)

    elif op == "erosi":
        return cv2.erode(gray, kernel, iterations=cfg.morph_iterations)

    elif op == "opening":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel,
                                iterations=cfg.morph_iterations)

    elif op == "closing":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel,
                                iterations=cfg.morph_iterations)

    elif op == "gradient":
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    elif op == "blur":
        k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
        return cv2.GaussianBlur(gray, (k, k), 0)

    elif op == "otsu":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    elif op == "pipeline":
        # Canny → Dilasi → Closing (pipeline lengkap)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges   = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        closed  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        return closed

    else:
        raise ValueError(f"Operasi tidak dikenal: {op}")


# ═══════════════════════════════════════════════════
# PARALLEL PROCESSOR UTAMA
# ═══════════════════════════════════════════════════

class ParallelImageProcessor:
    """
    Kelas utama: memproses banyak gambar secara paralel
    menggunakan multiprocessing.Pool.

    Penggunaan:
        processor = ParallelImageProcessor(n_workers=4)
        results = processor.process(image_paths, config)
    """

    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or cpu_count()
        self.n_workers = max(1, min(self.n_workers, cpu_count()))

    def process(self,
                image_paths: List[str],
                config: ProcessConfig,
                progress_callback=None) -> Tuple[List[ImageResult], float]:
        """
        Proses semua gambar secara paralel.

        Returns:
            (list of ImageResult, total_elapsed_seconds)
        """
        if not image_paths:
            return [], 0.0

        # Buat args untuk setiap worker
        cfg_dict = config.__dict__.copy()
        args = [(p, cfg_dict, i) for i, p in enumerate(image_paths)]

        t_start = time.perf_counter()
        results = []

        # ── multiprocessing.Pool ──────────────────
        # Pool membagi args ke n_workers proses secara otomatis
        with Pool(processes=self.n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_single_image, args)
            ):
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(image_paths), result)

        total_elapsed = time.perf_counter() - t_start
        return results, total_elapsed

    def process_sequential(self,
                           image_paths: List[str],
                           config: ProcessConfig,
                           progress_callback=None) -> Tuple[List[ImageResult], float]:
        """
        Proses gambar secara SEQUENTIAL (tanpa paralel).
        Digunakan untuk perbandingan benchmark.
        """
        cfg_dict = config.__dict__.copy()
        t_start = time.perf_counter()
        results = []

        for i, path in enumerate(image_paths):
            result = _process_single_image((path, cfg_dict, i))
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(image_paths), result)

        total_elapsed = time.perf_counter() - t_start
        return results, total_elapsed

    @staticmethod
    def benchmark(image_paths: List[str],
                  config: ProcessConfig,
                  max_workers: int = None) -> dict:
        """
        Benchmark: bandingkan waktu sequential vs paralel.
        Returns dict dengan statistik speedup.
        """
        max_w = max_workers or cpu_count()
        processor = ParallelImageProcessor()

        # Sequential
        print("[Benchmark] Sequential...")
        _, t_seq = processor.process_sequential(image_paths, config)

        # Paralel
        print(f"[Benchmark] Paralel ({max_w} workers)...")
        processor.n_workers = max_w
        _, t_par = processor.process(image_paths, config)

        speedup = t_seq / t_par if t_par > 0 else 0

        return {
            "n_images":        len(image_paths),
            "n_workers":       max_w,
            "time_sequential": t_seq,
            "time_parallel":   t_par,
            "speedup":         speedup,
            "efficiency":      speedup / max_w * 100,
        }
