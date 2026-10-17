"""exp_f.h's exp_f gives the C library's expf bits for every float.

On MinGW-w64 (msvcrt) x86-64, expf is (float)exp((double)x) and libmingwex's exp truncates
twice under a control word switched to chopping and back (two FLDCW, whose cost on a Zen 4
depends on where the linker puts the library). exp_f.h does the library's x87 sequence
with FISTTP + FILD for the truncations. This builds tests/exp_f_check.c as the engines are
built (-O3, an x86-64-v3 target, OpenMP) and compares the two on all 2^32 floats; with
msvcrt it must be the x87 exp_f. Elsewhere (UCRT, other C libraries) exp_f is expf itself.
"""
import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent


def gnu_gcc():
    gcc = shutil.which("gcc")
    if not gcc:
        return None
    version = subprocess.run([gcc, "--version"], capture_output=True, text=True).stdout
    return gcc if "clang" not in version.lower() else None


class ExpF(unittest.TestCase):
    def test_every_float(self):
        gcc = gnu_gcc()
        if not gcc:
            self.skipTest("GNU gcc not on PATH")
        if platform.machine().lower() not in ("amd64", "x86_64"):
            self.skipTest("x86-64 only: elsewhere exp_f is expf")
        static = ["-static"] if os.name == "nt" else []
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "exp_f_check.exe"
            subprocess.run([gcc, "-O3", "-march=x86-64-v3", "-fopenmp", "-Wall", str(C_DIR / "tests" / "exp_f_check.c"),
                            "-o", str(exe), *static, "-lm"], check=True)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=1800)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        lines = run.stdout.splitlines()
        self.assertEqual(lines[-1], "ok", run.stdout)
        self.assertIn(" 4294967296 floats, 0 differ", run.stdout)
        if "(msvcrt)" in lines[0]:   # MinGW on msvcrt: expf is libmingwex's x87 exp
            self.assertTrue(lines[0].startswith("x87 "), run.stdout)


if __name__ == "__main__":
    unittest.main()
