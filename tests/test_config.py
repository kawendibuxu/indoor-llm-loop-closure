from pathlib import Path

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
    assert config.output.output_dir == Path("/tmp/out")
    assert config.graph.max_relations_per_node == 4
