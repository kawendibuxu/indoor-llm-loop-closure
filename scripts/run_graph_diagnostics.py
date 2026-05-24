import argparse
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.diagnostics import summarize_graph_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()

    summary = summarize_graph_outputs(args.output_root, sample_count=args.sample_count)
    summary_path = args.output_root / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
