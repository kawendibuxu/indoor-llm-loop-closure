from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path


class OutputConfig(BaseModel):
    output_dir: Path


class ModelConfig(BaseModel):
    oneformer_model_name: str
    relabel_model_name: str


class GraphConfig(BaseModel):
    near_distance_m: float
    far_distance_m: float
    max_relations_per_node: int


class AppConfig(BaseModel):
    data: DataConfig
    output: OutputConfig
    models: ModelConfig
    graph: GraphConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)
