#!/usr/bin/env python3
"""Pack the Windows release of MiMo-V2.6-Pro on an RTX 4090 (docs/mimo-cuda.md).

Build first, from the repository root (MinGW-w64 GCC for the engine, Visual Studio
and the CUDA toolkit for the DLL):

    c\\build_mimo_cuda.bat
    mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
    python c/tools/package_mimo_4090.py --tag mimo-pro-4090-r1 --output dist

The archive holds the engine and its CUDA DLL, `coli` with every Python module it
reaches (tools/pack_python.py, as the main release does), the web UI, the measured
launcher (tools/start_mimo_cuda_4090.ps1, START-MIMO.cmd), its README and the
licenses, plus BUILD-INFO.txt: the commit and every binary's SHA-256.
"""
import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
C = ROOT / "c"

START_CMD = """@echo off
rem MiMo-V2.6-Pro on an RTX 4090: the measured profile (tools\\start_mimo_cuda_4090.ps1).
rem Arguments go to the script, e.g.  START-MIMO.cmd -ModelDir D:\\models\\MiMo-V2.6-Pro -ShardDir ""
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\\start_mimo_cuda_4090.ps1" %*
if errorlevel 1 pause
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="release name, e.g. mimo-pro-4090-r1")
    parser.add_argument("--output", default="dist", help="directory for the folder and the zip")
    args = parser.parse_args()

    for name in ("mimo.exe", "coli_cuda_mimo.dll"):
        if not (C / name).is_file():
            sys.exit(f"missing c/{name}: build it first (see this script's docstring)")
    out = pathlib.Path(args.output).resolve()
    stage = out / f"colibri-{args.tag}-windows-x86_64"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "tools").mkdir(parents=True)

    for name in ("mimo.exe", "coli_cuda_mimo.dll", "coli", "coli.cmd"):
        shutil.copy2(C / name, stage / name)
    subprocess.run([sys.executable, str(C / "tools" / "pack_python.py"), str(C), str(stage)], check=True)
    for name in ("start_mimo_cuda_4090.ps1", "benchmark_mimo_cuda.py", "check_mimo_cuda.py"):
        shutil.copy2(C / "tools" / name, stage / "tools" / name)
    (stage / "START-MIMO.cmd").write_bytes(START_CMD.replace("\n", "\r\n").encode("ascii"))
    shutil.copy2(ROOT / "docs" / "mimo-4090-release.md", stage / "README-MIMO-4090.md")
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, stage / name)
    web = ROOT / "web" / "dist"
    if web.is_dir():
        shutil.copytree(web, stage / "web" / "dist")

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    lines = [f"release {args.tag}", f"commit {commit}{' (with local changes)' if dirty else ''}",
             "mimo.exe: mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0 (needs AVX-512: Zen 4 or newer)",
             "coli_cuda_mimo.dll: c\\build_mimo_cuda.bat (sm_89, the CUDA runtime linked statically)", ""]
    for p in sorted(stage.rglob("*")):
        if p.is_file() and p.suffix in (".exe", ".dll"):
            lines.append(f"{sha256(p)}  {p.relative_to(stage).as_posix()}")
    (stage / "BUILD-INFO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive = out / f"{stage.name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, pathlib.Path(stage.name) / p.relative_to(stage))
    print(f"{archive} ({archive.stat().st_size / 1e6:.1f} MB), sha256 {sha256(archive)}")


if __name__ == "__main__":
    main()
