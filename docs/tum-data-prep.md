# TUM RGB-D Data Prep

这个项目不会替换掉 ScanNet 路径，只是额外支持 TUM/OpenLORIS 这一类 `RGB + depth + shared intrinsics` 的数据组织方式。

## TUM 基本思路

TUM 通常提供：

- `rgb/`
- `depth/`
- `associate.txt`

我们需要额外准备一个内参文件，例如：

```bash
python scripts/prepare_tum_sequence.py \
  --output-intrinsics /path/to/tum/intrinsics.txt
```

默认会写入：

```text
525.0 0.0 319.5
0.0 525.0 239.5
0.0 0.0 1.0
```

## 与当前批处理接口对接

当前 `run_sequence_graph.py` 仍然吃三目录输入：

- `color-dir`
- `depth-dir`
- `intrinsics-dir`

TUM 现在也支持单独脚本：

```bash
python scripts/run_tum_sequence_graph.py \
  --config configs/tum_sample.yaml \
  --associate /path/to/associate.txt \
  --dataset-root /path/to/tum_dataset \
  --intrinsics /path/to/tum/intrinsics.txt
```

这条路径仍然复用同一个 `process_frame_graph()`，不会影响 ScanNet 入口。
