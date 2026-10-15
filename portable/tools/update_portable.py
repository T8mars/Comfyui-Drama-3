"""Check the official release before launching, without importing Torch/ComfyUI."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drama3_t8.updater import check_and_update, RecoveryError, UpdateBusy, WorkspaceInUse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Drama 3 便携整合包代码更新；保留模型、Python和用户数据")
    parser.add_argument("--automatic", action="store_true", help="启动前检查；离线或普通更新失败继续使用原版本")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    def message(text):
        print(text, flush=True)
    try:
        message("正在检查 Drama 3 官方代码更新…")
        check_and_update(args.root, message)
        return 0
    except UpdateBusy as exc:
        message(f"更新或启动已由另一个窗口处理：{exc}")
        return 75
    except RecoveryError as exc:
        message(f"恢复未完成，已停止启动：{exc}")
        return 2
    except WorkspaceInUse as exc:
        message(f"本次未更新：{exc}")
        return 0 if args.automatic else 1
    except Exception as exc:
        message(f"本次未更新，继续保留当前版本：{exc}")
        return 0 if args.automatic else 1


if __name__ == "__main__":
    raise SystemExit(main())
