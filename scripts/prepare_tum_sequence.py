import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.io.tum import write_shared_intrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-intrinsics", type=Path, required=True)
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    args = parser.parse_args()

    write_shared_intrinsics(args.output_intrinsics, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    print(f"Wrote {args.output_intrinsics}")


if __name__ == "__main__":
    main()
