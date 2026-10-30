"""Build the full bundle's Windows x64 .NET Framework console entry point."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def _compiler() -> Path:
    windows = Path(os.environ.get("WINDIR", "C:/Windows"))
    return windows / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"


def _check_x64_console(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise RuntimeError("Compiler did not produce a Windows executable")
        offset = struct.unpack_from("<I", header, 60)[0]
        stream.seek(offset)
        pe = stream.read(96)
    if (len(pe) < 96 or pe[:4] != b"PE\0\0" or struct.unpack_from("<H", pe, 4)[0] != 0x8664
            or struct.unpack_from("<H", pe, 24)[0] != 0x20B
            or struct.unpack_from("<H", pe, 24 + 68)[0] != 3):
        raise RuntimeError("Compiler did not produce a Windows x64 console executable")


def build_launcher(output: Path, source: Path | None = None, compiler: Path | None = None) -> Path:
    """Compile atomically; a failed build never replaces an existing launcher."""
    output = Path(output).resolve()
    source = Path(source).resolve() if source is not None else Path(__file__).with_name("StartDrama3.cs").resolve()
    compiler = Path(compiler).resolve() if compiler is not None else _compiler()
    if output.suffix.lower() != ".exe":
        raise ValueError("Launcher output must end in .exe")
    if not source.is_file():
        raise FileNotFoundError(f"Missing launcher source: {source}")
    if not compiler.is_file():
        raise FileNotFoundError(f"Windows .NET Framework x64 C# compiler not found: {compiler}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".drama-launcher-", suffix=".exe", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        result = subprocess.run([str(compiler), "/nologo", "/target:exe", "/platform:x64", "/optimize+",
            "/debug-", "/codepage:65001", "/utf8output", f"/out:{temporary}", str(source)],
            shell=False, capture_output=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(f"Launcher compilation failed ({result.returncode}):\n" + (result.stdout + result.stderr)[-12000:])
        _check_x64_console(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Start-Drama3.exe")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--compiler", type=Path)
    options = parser.parse_args(argv)
    target = build_launcher(options.output, options.source, options.compiler)
    print(f"LAUNCHER_EXE {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
