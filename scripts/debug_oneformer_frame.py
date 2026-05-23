import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.config import load_config
from indoor_loop.io.scannet import load_scannet_frame
from indoor_loop.models.oneformer import run_oneformer_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rgb-path", type=Path, required=True)
    parser.add_argument("--depth-path", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_scannet_frame(args.rgb_path, args.depth_path, args.intrinsics)
    output = run_oneformer_with_fallback(
        frame.rgb,
        model_name=config.models.oneformer_model_name,
        use_demo=config.models.use_demo_oneformer,
    )
    print(f"run_mode={output.run_mode}")
    print(f"segment_count={len(output.segments)}")
    if output.error_message is not None:
        print(f"error={output.error_message}")


if __name__ == "__main__":
    main()
