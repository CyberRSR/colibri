"""Opt-in engine fallback tests: MIMO_CUDA_TEST=1 python this_file.py.

Requires the CUDA host build, tiny fixture and an NVIDIA GPU. Missing DLL,
unavailable device, zero budget and partial placement must preserve the oracle.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

C = Path(__file__).resolve().parents[1]
ENGINE = C / ("mimo.exe" if os.name == "nt" else "mimo")


@unittest.skipUnless(os.environ.get("MIMO_CUDA_TEST") == "1", "explicit CUDA hardware test")
class FallbackTest(unittest.TestCase):
    def run_case(self, engine=ENGINE, **overrides):
        reference = json.loads((C / "mimo_tiny" / "ref.json").read_text())
        case = reference["cases"]["short"]
        env = dict(os.environ, **reference["engine_env"])
        env.update(MIMO_CUDA="1", MIMO_CUDA_DENSE_GB="20", COLI_VULKAN="0", COLI_GPU="0")
        env.update(overrides)
        result = subprocess.run([str(engine), str(C / "mimo_tiny"), "--ids",
                                 " ".join(map(str, case["prompt_ids"])), "--ngen", str(case["max_new_tokens"])],
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(map(int, result.stdout.split())), case["greedy_new_ids"])
        return result.stderr

    @unittest.skipUnless(os.name == "nt", "Windows DLL loading")
    def test_missing_dll(self):
        with tempfile.TemporaryDirectory(prefix="mimo-cuda-loader-") as tmp:
            exe = Path(tmp) / "mimo.exe"
            shutil.copyfile(ENGINE, exe)
            self.assertIn("cannot load coli_cuda_mimo.dll", self.run_case(exe))

    def test_invalid_device(self):
        self.assertIn("mimo unavailable", self.run_case(COLI_GPU="9999"))

    def test_zero_budget(self):
        self.assertNotIn("mimo device", self.run_case(MIMO_CUDA_DENSE_GB="0"))

    def test_partial_placement(self):
        log = self.run_case(MIMO_DENSE_BITS="0", MIMO_CUDA_DENSE_GB="0.0001")
        self.assertIn("placed 1 dense matrices", log)
        self.assertRegex(log, r"[1-9][0-9]* matmuls")
        self.assertIn("0 scratch/compute fallbacks", log)


if __name__ == "__main__":
    unittest.main()
