import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.config import load_config
from indoor_loop.io.tum import collect_tum_triplets
from indoor_loop.pipeline import process_frame_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--associate", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    triplets = collect_tum_triplets(args.associate, args.dataset_root, args.intrinsics)
    for rgb_path, depth_path, intrinsics_path in triplets:
        stem = rgb_path.stem
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
