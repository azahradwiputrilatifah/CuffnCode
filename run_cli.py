"""
run_cli.py
==========
IFB 206 Komputasi Paralel | Evaluasi 3 | ITENAS Bandung

Mode CLI (tanpa GUI) untuk ParallelVision.
Berguna untuk demo di terminal / screen recording.

Penggunaan:
  python run_cli.py --folder sample_images --op canny --workers 4
  python run_cli.py --folder sample_images --op pipeline --workers 4 --benchmark
"""

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import cpu_count, freeze_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from parallel_processor import ParallelImageProcessor, ProcessConfig

COLORS = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "blue":   "\033[94m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(text, color): return f"{COLORS[color]}{text}{COLORS['reset']}"

def print_banner():
    print(c("=" * 60, "blue"))
    print(c("  ⚡ ParallelVision – CLI Mode", "cyan"))
    print(c("  IFB 206 Komputasi Paralel | Evaluasi 3 | ITENAS Bandung", "blue"))
    print(c("=" * 60, "blue"))
    print()

def progress_cb(cur, total, result):
    bar_len = 30
    filled  = int(bar_len * cur / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    status  = c("✓", "green") if result.success else c("✗", "red")
    name    = Path(result.path).name[:20].ljust(20)
    ms      = f"{result.elapsed_ms:6.1f}ms"
    print(f"\r  [{bar}] {cur:3}/{total}  {status}  {name}  {ms}", end="", flush=True)
    if cur == total:
        print()

def run_process(paths, config, n_workers, mode="parallel"):
    processor = ParallelImageProcessor(n_workers)
    t_label = c("PARALEL", "cyan") if mode == "parallel" else c("SEQUENTIAL", "yellow")
    print(f"  Mode: {t_label} | Workers: {c(str(n_workers), 'bold')} | Op: {c(config.operation.upper(), 'cyan')}")
    print()

    if mode == "parallel":
        results, elapsed = processor.process(paths, config, progress_cb)
    else:
        results, elapsed = processor.process_sequential(paths, config, progress_cb)

    ok    = sum(1 for r in results if r.success)
    t_per = sum(r.elapsed_ms for r in results) / max(1, len(results))
    thru  = len(results) / elapsed

    print()
    print(c(f"  ✅ Selesai: {ok}/{len(results)} gambar berhasil", "green"))
    print(f"  ⏱  Total   : {c(f'{elapsed:.3f}s', 'yellow')}")
    print(f"  ⏱  Rata-rata: {c(f'{t_per:.1f}ms', 'yellow')} per gambar")
    print(f"  🚀 Throughput: {c(f'{thru:.1f}', 'cyan')} gambar/detik")
    print(f"  💾 Output   : {c(config.output_dir, 'blue')}")
    return results, elapsed

def run_benchmark(paths, config):
    print(c("\n  📊 BENCHMARK: Sequential vs Paralel", "bold"))
    print(c("  " + "-" * 50, "blue"))
    processor = ParallelImageProcessor()

    # Sequential
    print(f"\n  {c('1.', 'yellow')} Sequential...")
    _, t_seq = processor.process_sequential(paths, config)
    print(f"     Waktu: {c(f'{t_seq:.3f}s', 'yellow')}")

    # Paralel berbagai worker
    results_par = {}
    for w in range(1, cpu_count() + 1):
        print(f"\n  {c(str(w)+'.', 'cyan')} Paralel {w} worker(s)...")
        processor.n_workers = w
        _, t = processor.process(paths, config)
        speedup = t_seq / t
        results_par[w] = (t, speedup)
        print(f"     Waktu: {c(f'{t:.3f}s', 'cyan')}  Speedup: {c(f'{speedup:.2f}×', 'green')}")

    # Ringkasan
    best_w = max(results_par, key=lambda w: results_par[w][1])
    best_t, best_sp = results_par[best_w]
    eff = best_sp / best_w * 100

    print()
    print(c("  ═══ RINGKASAN BENCHMARK ═══", "bold"))
    print(f"  Sequential    : {t_seq:.3f}s")
    for w, (t, sp) in results_par.items():
        bar = "▓" * int(sp * 5)
        print(f"  {w} worker(s)   : {t:.3f}s  Speedup {sp:.2f}×  {c(bar, 'green')}")
    print()
    print(c(f"  🏆 Best: {best_w} worker → speedup {best_sp:.2f}× (efisiensi {eff:.0f}%)", "green"))


def main():
    freeze_support()
    print_banner()

    parser = argparse.ArgumentParser(description="ParallelVision CLI")
    parser.add_argument("--folder",    default="sample_images", help="Folder gambar input")
    parser.add_argument("--output",    default="output",         help="Folder output")
    parser.add_argument("--op",        default="canny",
                        choices=["canny","dilasi","erosi","opening","closing","gradient","blur","otsu","pipeline"],
                        help="Operasi pemrosesan")
    parser.add_argument("--workers",   type=int, default=cpu_count(), help="Jumlah CPU worker")
    parser.add_argument("--clow",      type=int, default=50,   help="Canny threshold low")
    parser.add_argument("--chigh",     type=int, default=150,  help="Canny threshold high")
    parser.add_argument("--kernel",    type=int, default=5,    help="Ukuran kernel morfologi")
    parser.add_argument("--benchmark", action="store_true",    help="Jalankan benchmark")
    parser.add_argument("--sequential",action="store_true",    help="Mode sequential (untuk perbandingan)")
    args = parser.parse_args()

    # Load gambar
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    folder = args.folder
    if not os.path.isdir(folder):
        # coba relatif dari script
        folder = os.path.join(os.path.dirname(__file__), args.folder)

    if not os.path.isdir(folder):
        print(c(f"  ❌ Folder tidak ditemukan: {args.folder}", "red"))
        sys.exit(1)

    paths = sorted([str(p) for p in Path(folder).iterdir() if p.suffix.lower() in exts])
    if not paths:
        print(c(f"  ❌ Tidak ada gambar di: {folder}", "red"))
        sys.exit(1)

    print(f"  📂 Folder  : {c(folder, 'blue')}")
    print(f"  🖼  Gambar  : {c(str(len(paths)), 'cyan')}")
    print(f"  💻 CPU core: {c(str(cpu_count()), 'cyan')}")
    print()

    config = ProcessConfig(
        operation    = args.op,
        canny_low    = args.clow,
        canny_high   = args.chigh,
        morph_kernel = args.kernel,
        output_dir   = args.output,
        save_result  = True,
    )

    os.makedirs(args.output, exist_ok=True)

    if args.benchmark:
        run_benchmark(paths, config)
    elif args.sequential:
        run_process(paths, config, args.workers, "sequential")
    else:
        run_process(paths, config, args.workers, "parallel")

if __name__ == "__main__":
    main()
