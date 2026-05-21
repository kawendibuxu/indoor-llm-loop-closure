# ScanNet Data Prep

当前项目的批处理入口 `scripts/run_sequence_graph.py` 需要三目录结构：

- `color/`
- `depth/`
- `intrinsics/`

每一帧的命名约定是：

- `frame-000001.color.jpg`
- `frame-000001.depth.png`
- `frame-000001.intrinsics.txt`

## 如果你拿到的是 ScanNet 官方导出的 scene 目录

很多 scene 在导出后会长这样：

```text
scene0000_00/
  color/
    0.jpg
    1.jpg
  depth/
    0.png
    1.png
  intrinsic/
    intrinsic_depth.txt
```

这时直接运行：

```bash
python scripts/prepare_scannet_sequence.py \
  --source-root /path/to/scene0000_00 \
  --output-root /path/to/prepared/scene0000_00
```

它会转换成：

```text
/path/to/prepared/scene0000_00/
  color/
    frame-000000.color.jpg
  depth/
    frame-000000.depth.png
  intrinsics/
    frame-000000.intrinsics.txt
```

## 转换后批量跑图构建

```bash
python scripts/run_sequence_graph.py \
  --config configs/scannet_sample.yaml \
  --color-dir /path/to/prepared/scene0000_00/color \
  --depth-dir /path/to/prepared/scene0000_00/depth \
  --intrinsics-dir /path/to/prepared/scene0000_00/intrinsics
```

## 说明

- 当前脚本复用同一个 `intrinsic_depth.txt` 给所有帧
- 这符合当前 MVP 的输入假设
- 如果你后面要接更严格的 ScanNet 原始相机模型，再单独扩展
