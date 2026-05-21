import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.config import load_config
from indoor_loop.pipeline import process_frame_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    process_frame_graph(
        config.data.rgb_path,
        config.data.depth_path,
        config.data.intrinsics_path,
        config.output.output_dir,
        config,
    )
    output_path = config.output.output_dir / "scene_graph.json"
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
