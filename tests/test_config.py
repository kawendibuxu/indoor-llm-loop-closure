from pathlib import Path

import pytest
from pydantic import ValidationError

from indoor_loop.config import AppConfig, load_config


def test_load_config_reads_expected_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  rgb_path: /tmp/rgb.jpg",
                "  depth_path: /tmp/depth.png",
                "  intrinsics_path: /tmp/intrinsics.txt",
                "output:",
                "  output_dir: /tmp/out",
                "models:",
                "  oneformer_model_name: shi-labs/oneformer_ade20k_swin_large",
                "  relabel_model_name: google/siglip-base-patch16-224",
                "graph:",
                "  near_distance_m: 1.5",
                "  far_distance_m: 3.0",
                "  max_relations_per_node: 4",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.data.rgb_path == Path("/tmp/rgb.jpg")
    assert config.data.depth_path == Path("/tmp/depth.png")
    assert config.data.intrinsics_path == Path("/tmp/intrinsics.txt")
    assert config.output.output_dir == Path("/tmp/out")
    assert config.models.oneformer_model_name == "shi-labs/oneformer_ade20k_swin_large"
    assert config.graph.near_distance_m == 1.5
    assert config.graph.max_relations_per_node == 4


def test_load_config_reads_repository_sample_config() -> None:
    config = load_config(Path("configs/scannet_sample.yaml"))

    assert isinstance(config, AppConfig)
    assert config.data.rgb_path == Path("data/sample/frame-000100.color.jpg")
    assert config.data.depth_path == Path("data/sample/frame-000100.depth.png")
    assert config.data.intrinsics_path == Path("data/sample/intrinsics.txt")
    assert config.models.oneformer_model_name == "shi-labs/oneformer_ade20k_swin_large"
    assert config.graph.near_distance_m == 1.5


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  rgb_path: /tmp/rgb.jpg",
                "  depth_path: /tmp/depth.png",
                "  intrinsics_path: /tmp/intrinsics.txt",
                "  unexpected: nope",
                "output:",
                "  output_dir: /tmp/out",
                "models:",
                "  oneformer_model_name: shi-labs/oneformer_ade20k_swin_large",
                "  relabel_model_name: google/siglip-base-patch16-224",
                "graph:",
                "  near_distance_m: 1.5",
                "  far_distance_m: 3.0",
                "  max_relations_per_node: 4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)
