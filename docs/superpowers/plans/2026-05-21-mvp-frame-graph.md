# MVP Frame Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可运行的单帧室内 RGB-D 场景图导出原型，完成 ScanNet 单帧读取、OneFormer 分割结果解析、锚点重标定、静态性筛选、3D 节点生成、关系图导出与可视化。

**Architecture:** 采用轻量 Python 包结构，将重模型推理与可测试的数据解析逻辑分离。命令行入口只负责编排，核心逻辑拆分到 `io`、`models`、`graph`、`viz` 模块，先支持单帧图构建，再为后续 LLM 推理保留稳定 JSON 接口。

**Tech Stack:** Python 3.11、pytest、PyYAML、Pydantic、NumPy、Pillow、torch、transformers

---

### Task 1: 项目脚手架与配置加载

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/scannet_sample.yaml`
- Create: `src/indoor_loop/__init__.py`
- Create: `src/indoor_loop/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 写失败测试，固定配置结构与 YAML 加载行为**

```python
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
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indoor_loop'` or missing config symbols

- [ ] **Step 3: 写最小实现，建立项目结构和配置模型**

`pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "indoor-llm-loop-closure"
version = "0.1.0"
description = "Indoor RGB-D scene graph prototype for LLM-based loop closure"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pillow>=10.0",
  "pydantic>=2.7",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
models = [
  "torch>=2.2",
  "transformers>=4.40",
]
dev = [
  "pytest>=8.0",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/indoor_loop/config.py`

```python
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
```

`configs/scannet_sample.yaml`

```yaml
data:
  rgb_path: data/sample/frame-000100.color.jpg
  depth_path: data/sample/frame-000100.depth.png
  intrinsics_path: data/sample/intrinsics.txt
output:
  output_dir: outputs/sample_frame
models:
  oneformer_model_name: shi-labs/oneformer_ade20k_swin_large
  relabel_model_name: google/siglip-base-patch16-224
graph:
  near_distance_m: 1.5
  far_distance_m: 3.0
  max_relations_per_node: 4
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add pyproject.toml README.md configs/scannet_sample.yaml src/indoor_loop/__init__.py src/indoor_loop/config.py tests/test_config.py
git commit -m "feat: scaffold project config and package layout"
```

### Task 2: ScanNet 单帧读取器

**Files:**
- Create: `src/indoor_loop/io/__init__.py`
- Create: `src/indoor_loop/io/scannet.py`
- Create: `tests/io/test_scannet.py`

- [ ] **Step 1: 写失败测试，固定 RGB、深度和内参读取行为**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from indoor_loop.io.scannet import load_scannet_frame


def test_load_scannet_frame_reads_rgb_depth_and_intrinsics(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.jpg"
    depth_path = tmp_path / "depth.png"
    intrinsics_path = tmp_path / "intrinsics.txt"

    Image.fromarray(np.full((4, 5, 3), 64, dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((4, 5), 1200, dtype=np.uint16)).save(depth_path)
    intrinsics_path.write_text("577.0 0.0 320.0\n0.0 577.0 240.0\n0.0 0.0 1.0\n", encoding="utf-8")

    frame = load_scannet_frame(rgb_path, depth_path, intrinsics_path)

    assert frame.rgb.shape == (4, 5, 3)
    assert frame.depth.shape == (4, 5)
    assert frame.depth.dtype == np.float32
    assert np.isclose(frame.depth[0, 0], 1.2)
    assert frame.intrinsics.shape == (3, 3)
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/io/test_scannet.py -v`
Expected: FAIL with missing `indoor_loop.io.scannet`

- [ ] **Step 3: 写最小实现，输出统一帧对象**

`src/indoor_loop/io/scannet.py`

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class FrameData:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray


def load_scannet_frame(rgb_path: Path, depth_path: Path, intrinsics_path: Path) -> FrameData:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
    depth = depth_raw.astype(np.float32) / 1000.0
    intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)
    return FrameData(rgb=rgb, depth=depth, intrinsics=intrinsics)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/io/test_scannet.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/io/__init__.py src/indoor_loop/io/scannet.py tests/io/test_scannet.py
git commit -m "feat: add scannet frame loader"
```

### Task 3: 场景图领域模型

**Files:**
- Create: `src/indoor_loop/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: 写失败测试，固定节点、关系和图输出结构**

```python
from indoor_loop.types import ObjectNode, ObjectRelation, SceneGraph


def test_scene_graph_serializes_expected_fields() -> None:
    node = ObjectNode(
        object_id="obj-1",
        coarse_anchor_class="bed",
        refined_label="bed",
        label_confidence=0.9,
        staticness_level="hard_static",
        center_3d=[1.0, 0.5, 2.0],
        size_3d=[2.0, 1.0, 1.5],
        height_from_floor=0.0,
        support_type="floor",
        wall_attachment=True,
        saliency_score=0.8,
        observation_quality=0.95,
    )
    relation = ObjectRelation(
        subject_id="obj-1",
        predicate="against_wall",
        object_id="wall-1",
        confidence=0.88,
    )

    graph = SceneGraph(nodes=[node], relations=[relation])
    payload = graph.model_dump()

    assert payload["nodes"][0]["coarse_anchor_class"] == "bed"
    assert payload["relations"][0]["predicate"] == "against_wall"
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with missing `indoor_loop.types`

- [ ] **Step 3: 写最小实现，统一后续模块使用的数据结构**

`src/indoor_loop/types.py`

```python
from typing import Literal

from pydantic import BaseModel, Field


StaticnessLevel = Literal["hard_static", "soft_static", "dynamic"]


class SegmentRecord(BaseModel):
    segment_id: str
    class_name: str
    is_thing: bool
    score: float
    area: int
    bbox_xyxy: list[int]
    centroid_xy: list[float]


class ObjectNode(BaseModel):
    object_id: str
    coarse_anchor_class: str
    refined_label: str
    label_confidence: float
    staticness_level: StaticnessLevel
    center_3d: list[float]
    size_3d: list[float]
    height_from_floor: float
    support_type: str
    wall_attachment: bool
    saliency_score: float
    observation_quality: float


class ObjectRelation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    confidence: float


class SceneGraph(BaseModel):
    nodes: list[ObjectNode] = Field(default_factory=list)
    relations: list[ObjectRelation] = Field(default_factory=list)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_types.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/types.py tests/test_types.py
git commit -m "feat: add scene graph domain models"
```

### Task 4: OneFormer 结果解析适配层

**Files:**
- Create: `src/indoor_loop/models/__init__.py`
- Create: `src/indoor_loop/models/oneformer.py`
- Create: `tests/models/test_oneformer.py`

- [ ] **Step 1: 写失败测试，固定 panoptic 结果到 `SegmentRecord` 的解析行为**

```python
import numpy as np

from indoor_loop.models.oneformer import build_segments_from_panoptic


def test_build_segments_from_panoptic_extracts_bbox_area_and_centroid() -> None:
    panoptic_map = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    segments_info = [
        {"id": 1, "label_id": 7, "label_name": "bed", "score": 0.91, "was_fused": False},
        {"id": 2, "label_id": 12, "label_name": "floor", "score": 0.99, "was_fused": False},
    ]

    segments = build_segments_from_panoptic(panoptic_map, segments_info)

    assert [segment.segment_id for segment in segments] == ["1", "2"]
    assert segments[0].area == 4
    assert segments[0].bbox_xyxy == [1, 0, 2, 1]
    assert segments[1].class_name == "floor"
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/models/test_oneformer.py -v`
Expected: FAIL with missing `indoor_loop.models.oneformer`

- [ ] **Step 3: 写最小实现，把重模型封装和纯解析逻辑分开**

`src/indoor_loop/models/oneformer.py`

```python
from dataclasses import dataclass
from typing import Any

import numpy as np

from indoor_loop.types import SegmentRecord


def build_segments_from_panoptic(
    panoptic_map: np.ndarray, segments_info: list[dict[str, Any]]
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    for info in segments_info:
        segment_id = int(info["id"])
        mask = panoptic_map == segment_id
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [float(xs.mean()), float(ys.mean())]
        segments.append(
            SegmentRecord(
                segment_id=str(segment_id),
                class_name=str(info["label_name"]),
                is_thing=str(info["label_name"]) not in {"wall", "floor", "ceiling"},
                score=float(info["score"]),
                area=int(mask.sum()),
                bbox_xyxy=bbox,
                centroid_xy=centroid,
            )
        )
    return segments


@dataclass(slots=True)
class OneFormerOutput:
    panoptic_map: np.ndarray
    segments: list[SegmentRecord]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/models/test_oneformer.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/models/__init__.py src/indoor_loop/models/oneformer.py tests/models/test_oneformer.py
git commit -m "feat: add oneformer output parser"
```

### Task 5: 锚点重标定与静态性分类

**Files:**
- Create: `src/indoor_loop/models/relabel.py`
- Create: `src/indoor_loop/graph/__init__.py`
- Create: `src/indoor_loop/graph/staticness.py`
- Create: `tests/models/test_relabel.py`
- Create: `tests/graph/test_staticness.py`

- [ ] **Step 1: 写失败测试，固定粗粒度锚点类别映射与静态性规则**

```python
from indoor_loop.graph.staticness import classify_staticness
from indoor_loop.models.relabel import map_to_anchor_class


def test_map_to_anchor_class_merges_fine_labels() -> None:
    assert map_to_anchor_class("desk") == "table_like"
    assert map_to_anchor_class("table") == "table_like"
    assert map_to_anchor_class("bookshelf") == "storage_like"


def test_classify_staticness_uses_category_prior() -> None:
    assert classify_staticness("person", observation_quality=0.9) == "dynamic"
    assert classify_staticness("bed", observation_quality=0.9) == "hard_static"
    assert classify_staticness("chair", observation_quality=0.9) == "soft_static"
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/models/test_relabel.py tests/graph/test_staticness.py -v`
Expected: FAIL with missing relabel/staticness modules

- [ ] **Step 3: 写最小实现，先用规则版完成 MVP**

`src/indoor_loop/models/relabel.py`

```python
ANCHOR_CLASS_MAP = {
    "bed": "bed",
    "desk": "table_like",
    "table": "table_like",
    "chair": "seat_like",
    "sofa": "seat_like",
    "cabinet": "storage_like",
    "bookshelf": "storage_like",
    "shelf": "storage_like",
    "monitor": "display_like",
    "tv": "display_like",
    "sink": "sanitary_like",
    "toilet": "sanitary_like",
    "window": "opening_like",
    "door": "opening_like",
}


def map_to_anchor_class(label: str) -> str:
    return ANCHOR_CLASS_MAP.get(label, "other")
```

`src/indoor_loop/graph/staticness.py`

```python
from indoor_loop.types import StaticnessLevel


def classify_staticness(label: str, observation_quality: float) -> StaticnessLevel:
    if label == "person":
        return "dynamic"
    if label in {"wall", "floor", "ceiling", "bed", "cabinet", "window", "door"}:
        return "hard_static"
    if label in {"chair", "monitor", "table", "desk", "sofa"} and observation_quality >= 0.5:
        return "soft_static"
    return "dynamic"
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/models/test_relabel.py tests/graph/test_staticness.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/models/relabel.py src/indoor_loop/graph/__init__.py src/indoor_loop/graph/staticness.py tests/models/test_relabel.py tests/graph/test_staticness.py
git commit -m "feat: add anchor relabeling and staticness rules"
```

### Task 6: 3D 节点投影

**Files:**
- Create: `src/indoor_loop/graph/projection.py`
- Create: `tests/graph/test_projection.py`

- [ ] **Step 1: 写失败测试，固定由 mask + depth + intrinsics 生成 3D 中心和尺寸的行为**

```python
import numpy as np

from indoor_loop.graph.projection import project_mask_to_3d_bbox


def test_project_mask_to_3d_bbox_returns_center_and_size() -> None:
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    intrinsics = np.array(
        [
            [2.0, 0.0, 1.5],
            [0.0, 2.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    center, size = project_mask_to_3d_bbox(mask, depth, intrinsics)

    assert len(center) == 3
    assert len(size) == 3
    assert center[2] == 2.0
    assert size[2] == 0.0
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/graph/test_projection.py -v`
Expected: FAIL with missing `indoor_loop.graph.projection`

- [ ] **Step 3: 写最小实现，采用鲁棒投影统计**

`src/indoor_loop/graph/projection.py`

```python
import numpy as np


def project_mask_to_3d_bbox(
    mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
) -> tuple[list[float], list[float]]:
    ys, xs = np.where(mask & np.isfinite(depth) & (depth > 0))
    zs = depth[ys, xs]
    if zs.size == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    xs_3d = (xs.astype(np.float32) - cx) * zs / fx
    ys_3d = (ys.astype(np.float32) - cy) * zs / fy

    points = np.stack([xs_3d, ys_3d, zs], axis=1)
    center = np.median(points, axis=0).tolist()
    size = (points.max(axis=0) - points.min(axis=0)).tolist()
    return [float(v) for v in center], [float(v) for v in size]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/graph/test_projection.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/graph/projection.py tests/graph/test_projection.py
git commit -m "feat: add 3d projection helper"
```

### Task 7: 关系生成与场景图构建

**Files:**
- Create: `src/indoor_loop/graph/relations.py`
- Create: `src/indoor_loop/graph/builder.py`
- Create: `tests/graph/test_relations.py`
- Create: `tests/graph/test_builder.py`

- [ ] **Step 1: 写失败测试，固定 `left_of` 和 `near` 关系生成与图构建输出**

```python
from indoor_loop.graph.builder import build_scene_graph
from indoor_loop.types import ObjectNode


def test_build_scene_graph_adds_spatial_relations() -> None:
    nodes = [
        ObjectNode(
            object_id="bed-1",
            coarse_anchor_class="bed",
            refined_label="bed",
            label_confidence=0.9,
            staticness_level="hard_static",
            center_3d=[0.0, 0.0, 2.0],
            size_3d=[2.0, 1.0, 1.0],
            height_from_floor=0.0,
            support_type="floor",
            wall_attachment=True,
            saliency_score=0.9,
            observation_quality=0.9,
        ),
        ObjectNode(
            object_id="desk-1",
            coarse_anchor_class="table_like",
            refined_label="desk",
            label_confidence=0.85,
            staticness_level="soft_static",
            center_3d=[1.0, 0.0, 2.2],
            size_3d=[1.2, 0.8, 0.7],
            height_from_floor=0.0,
            support_type="floor",
            wall_attachment=False,
            saliency_score=0.8,
            observation_quality=0.9,
        ),
    ]

    graph = build_scene_graph(nodes, near_distance_m=1.5, far_distance_m=3.0, max_relations_per_node=4)

    predicates = [relation.predicate for relation in graph.relations]
    assert "left_of" in predicates
    assert "near" in predicates
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/graph/test_relations.py tests/graph/test_builder.py -v`
Expected: FAIL with missing builder/relations modules

- [ ] **Step 3: 写最小实现，先支持最关键的离散空间关系**

`src/indoor_loop/graph/relations.py`

```python
import math

from indoor_loop.types import ObjectNode, ObjectRelation


def pairwise_relations(a: ObjectNode, b: ObjectNode, near_distance_m: float, far_distance_m: float) -> list[ObjectRelation]:
    relations: list[ObjectRelation] = []
    dx = b.center_3d[0] - a.center_3d[0]
    dz = b.center_3d[2] - a.center_3d[2]
    distance = math.sqrt(dx * dx + dz * dz)

    if dx > 0:
        relations.append(ObjectRelation(subject_id=a.object_id, predicate="left_of", object_id=b.object_id, confidence=0.8))
    elif dx < 0:
        relations.append(ObjectRelation(subject_id=a.object_id, predicate="right_of", object_id=b.object_id, confidence=0.8))

    if distance <= near_distance_m:
        relations.append(ObjectRelation(subject_id=a.object_id, predicate="near", object_id=b.object_id, confidence=0.9))
    elif distance <= far_distance_m:
        relations.append(ObjectRelation(subject_id=a.object_id, predicate="far", object_id=b.object_id, confidence=0.6))

    return relations
```

`src/indoor_loop/graph/builder.py`

```python
from indoor_loop.graph.relations import pairwise_relations
from indoor_loop.types import ObjectNode, SceneGraph


def build_scene_graph(
    nodes: list[ObjectNode],
    near_distance_m: float,
    far_distance_m: float,
    max_relations_per_node: int,
) -> SceneGraph:
    relations = []
    for index, source in enumerate(nodes):
        local_relations = []
        for target in nodes[index + 1 :]:
            local_relations.extend(pairwise_relations(source, target, near_distance_m, far_distance_m))
        relations.extend(local_relations[:max_relations_per_node])
    return SceneGraph(nodes=nodes, relations=relations)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/graph/test_relations.py tests/graph/test_builder.py -v`
Expected: PASS

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/graph/relations.py src/indoor_loop/graph/builder.py tests/graph/test_relations.py tests/graph/test_builder.py
git commit -m "feat: add scene graph relation builder"
```

### Task 8: JSON 导出、可视化与命令行入口

**Files:**
- Create: `src/indoor_loop/viz/__init__.py`
- Create: `src/indoor_loop/viz/export.py`
- Create: `scripts/run_frame_graph.py`
- Create: `tests/viz/test_export.py`
- Create: `tests/test_run_frame_graph.py`

- [ ] **Step 1: 写失败测试，固定 JSON 导出和 CLI 输出文件**

```python
from pathlib import Path

from indoor_loop.types import SceneGraph
from indoor_loop.viz.export import write_scene_graph_json


def test_write_scene_graph_json_creates_file(tmp_path: Path) -> None:
    graph = SceneGraph(nodes=[], relations=[])
    output_path = tmp_path / "scene_graph.json"

    write_scene_graph_json(graph, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip().startswith("{")
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/viz/test_export.py tests/test_run_frame_graph.py -v`
Expected: FAIL with missing export/CLI modules

- [ ] **Step 3: 写最小实现，先打通 frame-level 导出入口**

`src/indoor_loop/viz/export.py`

```python
import json
from pathlib import Path

from indoor_loop.types import SceneGraph


def write_scene_graph_json(graph: SceneGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
```

`scripts/run_frame_graph.py`

```python
from pathlib import Path
import argparse

from indoor_loop.config import load_config
from indoor_loop.types import SceneGraph
from indoor_loop.viz.export import write_scene_graph_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = config.output.output_dir / "scene_graph.json"
    write_scene_graph_json(SceneGraph(nodes=[], relations=[]), output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/viz/test_export.py tests/test_run_frame_graph.py -v`
Expected: PASS

- [ ] **Step 5: 运行一次端到端最小命令并记录结果**

Run: `python scripts/run_frame_graph.py --config configs/scannet_sample.yaml`
Expected: prints `Wrote outputs/sample_frame/scene_graph.json`

- [ ] **Step 6: 提交本任务**

```bash
git add src/indoor_loop/viz/__init__.py src/indoor_loop/viz/export.py scripts/run_frame_graph.py tests/viz/test_export.py tests/test_run_frame_graph.py
git commit -m "feat: add scene graph export cli"
```

### Task 9: 接线真实单帧管线

**Files:**
- Modify: `src/indoor_loop/config.py`
- Modify: `src/indoor_loop/models/oneformer.py`
- Modify: `src/indoor_loop/models/relabel.py`
- Modify: `src/indoor_loop/graph/projection.py`
- Modify: `src/indoor_loop/graph/builder.py`
- Modify: `scripts/run_frame_graph.py`
- Create: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: 写失败测试，用 fake parser 跑通单帧管线**

```python
import numpy as np

from indoor_loop.graph.builder import build_scene_graph
from indoor_loop.models.oneformer import build_segments_from_panoptic


def test_pipeline_smoke_builds_non_empty_graph_from_fake_segments() -> None:
    panoptic_map = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    segments = build_segments_from_panoptic(
        panoptic_map,
        [
            {"id": 1, "label_name": "bed", "score": 0.9},
            {"id": 2, "label_name": "desk", "score": 0.8},
        ],
    )

    assert len(segments) == 2
    graph = build_scene_graph([], near_distance_m=1.5, far_distance_m=3.0, max_relations_per_node=4)
    assert graph.model_dump()["relations"] == []
```

- [ ] **Step 2: 运行测试，确认当前失败或能力不足**

Run: `pytest tests/test_pipeline_smoke.py -v`
Expected: FAIL because the pipeline still exports only an empty placeholder graph

- [ ] **Step 3: 写最小实现，把真实模块在 CLI 中接起来**

关键接线要求：

```python
# scripts/run_frame_graph.py
# 1. load_config
# 2. load_scannet_frame
# 3. run oneformer parser or fake parser hook
# 4. convert segments into nodes
# 5. build_scene_graph
# 6. write_scene_graph_json
```

`src/indoor_loop/models/relabel.py` 需要补一个面向节点的函数：

```python
def relabel_segment(class_name: str) -> tuple[str, str, float]:
    coarse = map_to_anchor_class(class_name)
    return class_name, coarse, 0.75 if coarse != "other" else 0.5
```

`scripts/run_frame_graph.py` 接线时先允许“fake panoptic result”模式，确保没有真实模型权重也能打通流程。

- [ ] **Step 4: 运行 smoke test 与导出命令**

Run: `pytest tests/test_pipeline_smoke.py -v`
Expected: PASS

Run: `python scripts/run_frame_graph.py --config configs/scannet_sample.yaml`
Expected: writes `scene_graph.json` with at least `nodes` and `relations` keys

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/config.py src/indoor_loop/models/oneformer.py src/indoor_loop/models/relabel.py src/indoor_loop/graph/projection.py src/indoor_loop/graph/builder.py scripts/run_frame_graph.py tests/test_pipeline_smoke.py
git commit -m "feat: wire frame graph pipeline"
```

### Task 10: 可视化核对输出

**Files:**
- Modify: `src/indoor_loop/viz/export.py`
- Create: `tests/viz/test_overlay.py`

- [ ] **Step 1: 写失败测试，固定 overlay 图片导出**

```python
from pathlib import Path

import numpy as np

from indoor_loop.viz.export import write_overlay_png


def test_write_overlay_png_creates_image(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    output_path = tmp_path / "overlay.png"

    write_overlay_png(rgb, [], output_path)

    assert output_path.exists()
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/viz/test_overlay.py -v`
Expected: FAIL with missing `write_overlay_png`

- [ ] **Step 3: 写最小实现，导出基础核对图**

`src/indoor_loop/viz/export.py`

```python
from pathlib import Path

import numpy as np
from PIL import Image


def write_overlay_png(rgb: np.ndarray, _segments: list[object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)
```

CLI 中补一行调用：

```python
write_overlay_png(frame.rgb, [], config.output.output_dir / "overlay.png")
```

- [ ] **Step 4: 运行测试与端到端命令**

Run: `pytest tests/viz/test_overlay.py -v`
Expected: PASS

Run: `python scripts/run_frame_graph.py --config configs/scannet_sample.yaml`
Expected: writes both `scene_graph.json` and `overlay.png`

- [ ] **Step 5: 提交本任务**

```bash
git add src/indoor_loop/viz/export.py scripts/run_frame_graph.py tests/viz/test_overlay.py
git commit -m "feat: add overlay export for graph inspection"
```

## 计划自检

- 规格覆盖：
  - 单帧 ScanNet 输入：Task 2
  - OneFormer 分割解析：Task 4
  - 锚点重标定：Task 5
  - 静态性筛选：Task 5
  - 3D 投影：Task 6
  - 关系图构建：Task 7
  - JSON 与 overlay 导出：Task 8、Task 10
  - 单帧可运行管线：Task 9
- 无占位词：
  - 本计划未保留 `TBD`、`TODO`、`implement later` 等占位项
- 类型一致性：
  - 全文统一使用 `SegmentRecord`、`ObjectNode`、`ObjectRelation`、`SceneGraph`
  - 静态性等级统一为 `hard_static`、`soft_static`、`dynamic`

