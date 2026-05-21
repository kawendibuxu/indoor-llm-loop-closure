from pathlib import Path

from indoor_loop.config import load_config
from indoor_loop.io.scannet import load_scannet_frame
from indoor_loop.models.oneformer import build_demo_oneformer_output


def test_sample_config_has_output_dir() -> None:
    config = load_config(Path("configs/scannet_sample.yaml"))
    assert config.output.output_dir == Path("outputs/sample_frame")


def test_demo_oneformer_output_matches_loaded_frame() -> None:
    config = load_config(Path("configs/scannet_sample.yaml"))
    frame = load_scannet_frame(
        config.data.rgb_path,
        config.data.depth_path,
        config.data.intrinsics_path,
    )

    output = build_demo_oneformer_output(frame.rgb)

    assert output.panoptic_map.shape == frame.depth.shape
    assert len(output.segments) == 2
