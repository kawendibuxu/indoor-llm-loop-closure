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

对于 TUM，后续建议再补一个专门的 `run_tum_sequence_graph.py`，直接读取 `associate.txt`。
当前阶段先把 TUM 适配保持在 `io` 层，不动上层场景图 pipeline。
