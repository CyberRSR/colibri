"""The benchmark must reject silent CPU fallback, but accept 110 GPU matrices."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_mimo_cuda.py"
spec = importlib.util.spec_from_file_location("benchmark_mimo_cuda", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class CudaEvidenceTest(unittest.TestCase):
    def test_actual_work(self):
        bench.verify_cuda("[CUDA] mimo dense: 110 matrices, 19.995 GiB resident, 28160 matmuls, 35136 rows, 0 scratch/compute fallbacks")

    def test_discovery_is_not_work(self):
        for text in ("[CUDA] mimo device 0: RTX 4090",
                     "[CUDA] mimo dense: 0 matrices, 0.000 GiB resident, 0 matmuls,",
                     "[CUDA] mimo dense: 110 matrices, 19.995 GiB resident, 0 matmuls,",
                     "[CUDA] mimo dense: 110 matrices, 19.995 GiB resident, 1 matmuls, device failed"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                bench.verify_cuda(text)


if __name__ == "__main__":
    unittest.main()
