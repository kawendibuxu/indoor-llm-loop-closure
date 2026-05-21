from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oneformer_model_name: str
    relabel_model_name: str
    use_demo_oneformer: bool = False


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    near_distance_m: float
    far_distance_m: float
    max_relations_per_node: int


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    output: OutputConfig
    models: ModelConfig
    graph: GraphConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)
