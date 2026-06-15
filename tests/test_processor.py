"""
test_processor.py
=================
IFB 206 Komputasi Paralel | Evaluasi 3 | ITENAS Bandung

Unit test untuk ParallelImageProcessor dan operasi citra.
Jalankan: python3 tests/test_processor.py
"""

import unittest
import os
import sys
import tempfile
import shutil
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from parallel_processor import (
    ParallelImageProcessor, ProcessConfig, ImageResult,
    _process_single_image, _apply_operation
)


def make_test_image(path, size=(128, 128)):
    """Buat gambar sintetis untuk test."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    cv2.circle(img, (64, 64), 40, (255, 255, 255), -1)
    cv2.rectangle(img, (10, 10), (50, 50), (0, 128, 255), 2)
    gauss = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


class TestApplyOperation(unittest.TestCase):
    """Test tiap operasi pemrosesan pada gambar grayscale."""

    def setUp(self):
        self.gray = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        self.cfg  = ProcessConfig()

    def test_canny_output_shape(self):
        result = _apply_operation(self.gray, "canny", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_dilasi_output_shape(self):
        result = _apply_operation(self.gray, "dilasi", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_erosi_output_shape(self):
        result = _apply_operation(self.gray, "erosi", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_opening_output_shape(self):
        result = _apply_operation(self.gray, "opening", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_closing_output_shape(self):
        result = _apply_operation(self.gray, "closing", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_gradient_output_shape(self):
        result = _apply_operation(self.gray, "gradient", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_blur_output_shape(self):
        result = _apply_operation(self.gray, "blur", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_otsu_binary_values(self):
        result = _apply_operation(self.gray, "otsu", self.cfg)
        unique = np.unique(result)
        self.assertTrue(all(v in [0, 255] for v in unique),
                        "Otsu harus menghasilkan gambar biner (0 dan 255)")

    def test_pipeline_output_shape(self):
        result = _apply_operation(self.gray, "pipeline", self.cfg)
        self.assertEqual(result.shape, self.gray.shape)

    def test_invalid_operation_raises(self):
        with self.assertRaises(ValueError):
            _apply_operation(self.gray, "operasi_tidak_ada", self.cfg)

    def test_canny_edges_are_binary(self):
        result = _apply_operation(self.gray, "canny", self.cfg)
        unique = np.unique(result)
        self.assertTrue(all(v in [0, 255] for v in unique),
                        "Canny harus menghasilkan nilai 0 atau 255")

    def test_dilasi_non_decreasing(self):
        """Dilasi tidak boleh mengecilkan nilai piksel maksimum."""
        result = _apply_operation(self.gray, "dilasi", self.cfg)
        self.assertGreaterEqual(int(result.max()), int(self.gray.max()) - 1)

    def test_erosi_non_increasing(self):
        """Erosi tidak boleh memperbesar nilai piksel minimum."""
        result = _apply_operation(self.gray, "erosi", self.cfg)
        self.assertLessEqual(int(result.min()), int(self.gray.min()) + 1)


class TestProcessSingleImage(unittest.TestCase):
    """Test worker function pada file gambar nyata."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = make_test_image(os.path.join(self.tmpdir, "test.jpg"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_successful_processing(self):
        cfg = ProcessConfig(output_dir=self.tmpdir, save_result=True)
        result = _process_single_image((self.img_path, cfg.__dict__, 0))
        self.assertTrue(result.success, f"Gagal: {result.error}")
        self.assertGreater(result.elapsed_ms, 0)

    def test_output_image_not_none(self):
        cfg = ProcessConfig(output_dir=self.tmpdir, save_result=False)
        result = _process_single_image((self.img_path, cfg.__dict__, 0))
        self.assertIsNotNone(result.output_image)

    def test_output_shape_is_3channel(self):
        cfg = ProcessConfig(output_dir=self.tmpdir, save_result=False)
        result = _process_single_image((self.img_path, cfg.__dict__, 0))
        self.assertEqual(len(result.output_image.shape), 3)

    def test_invalid_path_returns_failure(self):
        cfg = ProcessConfig(output_dir=self.tmpdir)
        result = _process_single_image(("/tidak/ada/gambar.jpg", cfg.__dict__, 0))
        self.assertFalse(result.success)
        self.assertNotEqual(result.error, "")

    def test_output_file_saved(self):
        cfg = ProcessConfig(output_dir=self.tmpdir, save_result=True, operation="canny")
        _process_single_image((self.img_path, cfg.__dict__, 0))
        saved = os.listdir(self.tmpdir)
        canny_files = [f for f in saved if "canny" in f]
        self.assertTrue(len(canny_files) > 0, "File output canny tidak ditemukan")


class TestParallelProcessor(unittest.TestCase):
    """Test kelas ParallelImageProcessor."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.paths = [
            make_test_image(os.path.join(self.tmpdir, f"img_{i}.jpg"))
            for i in range(5)
        ]
        self.config = ProcessConfig(
            output_dir=os.path.join(self.tmpdir, "out"),
            save_result=True
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parallel_returns_all_results(self):
        proc = ParallelImageProcessor(n_workers=2)
        results, elapsed = proc.process(self.paths, self.config)
        self.assertEqual(len(results), len(self.paths))

    def test_sequential_returns_all_results(self):
        proc = ParallelImageProcessor()
        results, elapsed = proc.process_sequential(self.paths, self.config)
        self.assertEqual(len(results), len(self.paths))

    def test_parallel_all_success(self):
        proc = ParallelImageProcessor(n_workers=2)
        results, _ = proc.process(self.paths, self.config)
        for r in results:
            self.assertTrue(r.success, f"Gagal: {r.error}")

    def test_parallel_elapsed_positive(self):
        proc = ParallelImageProcessor(n_workers=2)
        _, elapsed = proc.process(self.paths, self.config)
        self.assertGreater(elapsed, 0)

    def test_empty_list_returns_empty(self):
        proc = ParallelImageProcessor()
        results, elapsed = proc.process([], self.config)
        self.assertEqual(results, [])
        self.assertEqual(elapsed, 0.0)

    def test_n_workers_capped_at_cpu(self):
        from multiprocessing import cpu_count
        proc = ParallelImageProcessor(n_workers=9999)
        self.assertLessEqual(proc.n_workers, cpu_count())

    def test_progress_callback_called(self):
        calls = []
        def cb(cur, total, result):
            calls.append(cur)

        proc = ParallelImageProcessor(n_workers=2)
        proc.process(self.paths, self.config, progress_callback=cb)
        self.assertEqual(len(calls), len(self.paths))


if __name__ == "__main__":
    print("=" * 60)
    print("  ParallelVision – Unit Test")
    print("  IFB 206 Komputasi Paralel | ITENAS Bandung")
    print("=" * 60)
    unittest.main(verbosity=2)
