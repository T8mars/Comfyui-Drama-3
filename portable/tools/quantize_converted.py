"""Source-checkout compatibility alias; portable bundles use convert_model.py quantize."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_model import main as _convert_main


def main(argv=None):
    return _convert_main(["quantize", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
