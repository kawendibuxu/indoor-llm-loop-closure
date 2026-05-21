import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.config import load_config
from indoor_loop.io.scannet import collect_sequence_triplets
from indoor_loop.pipeline import process_frame_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--color-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--intrinsics-dir", type=Path, required=True)
    args = parser.parse_args()

    load_config(args.config)
    config = load_config(args.config)
    triplets = collect_sequence_triplets(args.color_dir, args.depth_dir, args.intrinsics_dir)
    for rgb_path, depth_path, intrinsics_path in triplets:
        stem = rgb_path.name.replace(".color.jpg", "")
        process_frame_graph(
            rgb_path,
            depth_path,
            intrinsics_path,
            config.output.output_dir / stem,
            config,
        )
    print(f"Found {len(triplets)} frame triplets")


if __name__ == "__main__":
    main()
