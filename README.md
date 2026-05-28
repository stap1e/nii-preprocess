# Medical CT Preprocessing

用于腹部 CT 半监督/监督分割的数据预处理脚本集合，覆盖 **FLARE**、**AMOS**、**AbdomenCT-1K**、**Synapse** 等数据集。

原先散落在仓库根目录的脚本已按职责拆分，公共逻辑集中在 `med_preprocess` 包中，便于复用与扩展。

## 目录结构

```
med_preprocess/          # 可复用的 Python 库
  io/                    # NIfTI / H5 读写
  transforms/            # 重采样、强度归一化
  splits/                # 划分 train/val/test 列表

scripts/
  cli/                   # 通用命令行入口
  datasets/
    flare/               # FLARE25 相关
    amos/                # AMOS22 相关
    abdomen_ct/          # AbdomenCT-1K 相关
    synapse/             # Synapse 相关
  tools/                 # 检查、重命名、移动等工具脚本
  legacy/                # 旧版一键脚本（保留行为，依赖 med_preprocess）

configs/
  paths.example.yaml     # 路径配置示例（复制为 paths.local.yaml 后修改）
```

## 安装

```bash
cd /path/to/repo
pip install -e .
# 若需要 MONAI 流水线：
pip install -e ".[monai]"
```

## 通用 CLI

### NIfTI → H5

```bash
python scripts/cli/nii_to_h5.py \
  --image-dir /data/images \
  --mask-dir /data/labels \
  --output-dir /data/h5

# 无标签数据
python scripts/cli/nii_to_h5.py \
  --image-dir /data/unlabeled \
  --output-dir /data/h5_u \
  --unlabeled

# Synapse 命名（img0001.nii ↔ label0001.nii）
python scripts/cli/nii_to_h5.py \
  --image-dir /data/img \
  --mask-dir /data/label \
  --output-dir /data/h5 \
  --synapse-naming
```

也可在代码中调用：

```python
from med_preprocess.io import H5Converter

H5Converter(
    image_dir="/data/images",
    mask_dir="/data/labels",
    output_dir="/data/h5",
).run()
```

### 生成分割列表

```bash
python scripts/cli/generate_splits.py \
  --data-dir /data/labeled_h5 \
  --extension h5 \
  --id-regex '_(\d+)' \
  --split train_l=160 --split val=20 --split test=20 \
  --output train_l=/data/train_l.txt \
  --output val=/data/val.txt \
  --output test=/data/test.txt
```

各数据集原有的 `getsplit*.py` 仍保留在 `scripts/datasets/<name>/`，内含具体划分规模与路径，可按需逐步迁到 YAML 配置。

## 数据集脚本（原入口对照）

| 原根目录文件 | 新位置 |
|-------------|--------|
| `preprocess_flare.py` | `scripts/datasets/flare/preprocess_flare.py` |
| `niitoh5.py`, `savenii2h5.py` | `scripts/datasets/flare/` |
| `preprocess_amos.py` | `scripts/datasets/amos/preprocess_amos.py` |
| `AK_process.py`, `getsplit_ak.py` | `scripts/datasets/abdomen_ct/` |
| `synapse_preprocess.py` 等 | `scripts/datasets/synapse/` |
| `check_nii.py` | `scripts/tools/check_nii.py` |
| `preprocess_notuse.py` | `scripts/legacy/preprocess_notuse.py` |

从仓库根目录运行示例：

```bash
python scripts/datasets/flare/preprocess_flare.py
```

## 扩展新数据集

1. 在 `scripts/datasets/<your_dataset>/` 添加预处理脚本。
2. 复用 `med_preprocess.io.H5Converter` 与 `med_preprocess.transforms` 中的函数，避免复制 NIfTI/H5 逻辑。
3. 在 `configs/paths.example.yaml` 增加一节路径说明；本地复制为 `paths.local.yaml`（已 gitignore）。
4. 若划分规则固定，可封装 `med_preprocess.splits.SplitPlan` + `write_split_lists`，或扩展 `scripts/cli/generate_splits.py`。

## 说明

- 多数脚本内仍含 **Windows 绝对路径**（`E:/...`），与历史实验环境一致；新实验建议逐步改为读取 `configs/paths.local.yaml` 或通过 CLI 传参。
- `scripts/legacy/` 保留旧 FLARE 批处理流程，核心算法已迁入 `med_preprocess`。
