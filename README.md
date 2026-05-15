# MedSeg

本仓库提供一个面向“检索增强分割”的最小工程化实现：输入待测图片，通过 RAG 系统检索 Top-k 相似样例（带专家标注掩码），将这些样例写入 SAM3 的 memory bank，并在锁定 memory bank 后对目标帧进行分割，输出二值掩码图像文件。

## 功能概览

- RAG 检索（DINOv3 embedding + FAISS）：输入查询图片路径，返回 Top-k `[(img_path, mask_path, distance), ...]`
- SAM3 记忆注意力分割：把 Top-k 样例作为“条件帧”写入 memory bank，把待测图片作为“目标帧”推理并输出掩码
- memory bank 锁定：Top-k 写入完成后对目标帧推理禁用 memory encoder（语义等价 `run_mem_encoder=False`），避免记忆被目标帧污染
- 产物输出：`*_mask.png`（单通道 8-bit，0/255）与对应的 `*.json` 元信息
- 可视化叠加：从 `*.json` 自动读取原图与 mask，生成 alpha 混合叠加图（更直观对比）

## 目录结构

- `rag/`：检索系统实现（embedding、索引、mask 元数据存储、数据集导入）
- `seg_pipeline/`：分割流水线（输入校验、RAG 对接、SAM3 适配、输出落盘）
- `scripts/`：辅助脚本（SAM3 权重下载等）
- `resource/sam3/`：SAM3 源码与文档（上游工程内置）
- `resource/dinov3/`、`resource/faiss/`：上游依赖源码（供参考/对照）
- `segment.md`：分割需求说明
- `rag.md`：RAG 需求说明
- `.trae/specs/`：spec-driven 开发产物（spec/tasks/checklist）

## 关键接口

### RAG：检索 Top-k

入口类：`rag.rag_system.RAGSystem`

- `search(query_image: str, k: int=None) -> List[Tuple[str, str, float]]`
  - 返回：`[(img_path, mask_path, distance), ...]`
  - `mask_path` 可能为空字符串（未配置对应 mask）

RAG 也提供索引维护接口：`index_images` / `add_image` / `remove_image` / `clear_index`。

#### 数据集导入（`rag.dataset_importers`）

支持从通用视觉分割数据集格式直接导入图像-掩码对，无需手动逐条添加。

**支持格式：**

| 格式 | 标识符 | 输入 | 说明 |
|------|--------|------|------|
| COCO JSON | `"coco"` | 标注 JSON 文件路径 | 支持 polygon / 非压缩 RLE / 压缩 RLE 三种分割标注，可选类别过滤 |
| Pascal VOC | `"voc"` | 数据集根目录 | 从 ImageSets 或 SegmentationClass 反向配对，支持按类别 ID/名称提取 |

**RAGSystem 集成：**

```python
from rag import RAGSystem

rag = RAGSystem()
rag.import_dataset("coco", "/path/to/annotations/instances.json",
                   image_dir="/path/to/images/",
                   category_ids=[1, 3])           # 可选：仅导入指定类别
rag.import_dataset("voc", "/path/to/VOC2012/",
                   class_name="cat")               # 可选：仅提取指定类别
```

**独立使用 importer：**

```python
from rag.dataset_importers import get_importer, list_formats

print(list_formats())  # ["coco", "voc"]

importer = get_importer("coco")
pairs = importer.discover_pairs("annotations.json", image_dir="images/")
# 返回 [(image_abs_path, mask_abs_path), ...]
```

**注册自定义格式：**

```python
from rag.dataset_importers import register_importer, DatasetImporter

class CityscapesImporter(DatasetImporter):
    def discover_pairs(self, dataset_path, **kwargs):
        ...  # 返回 [(image_path, mask_path), ...]

register_importer("cityscapes", CityscapesImporter)
```

### 分割流水线：segment_image

入口函数：`seg_pipeline.segment_image`

```python
from seg_pipeline import segment_image
from seg_pipeline.sam3_memory_segmenter import Sam3BuildConfig

result = segment_image(
    query_image_path="/abs/path/query.png",
    k=5,  # 或者直接传 retrieval_topk=[(...), ...]
    sam3_build_config=Sam3BuildConfig(
        version="sam3.1",  # "sam3" 或 "sam3.1"
        checkpoint_path="/abs/path/ckpt.pt",  # 可选；不填则由 SAM3 builder 自行处理
        bpe_path=None,
    ),
    output_prob_thresh=0.5,  # 二值化阈值（概率）
    lock_memory=True,  # Top-k 写入后锁定 memory bank（推荐）
)

print(result.output_mask_path)  # 例如 .../outputs/query_mask.png
print(result.meta_path)         # 例如 .../outputs/query_mask.png.json
```

### 可视化：alpha 混合叠加（推荐）

分割输出 mask 有时与原图分辨率不一致。项目提供了一个“从 meta 自动对齐并叠加”的便捷函数：

```python
from seg_pipeline import overlay_from_meta

vis = overlay_from_meta(result.meta_path, alpha=0.45, color=(255, 0, 0))
vis.show()  # 或在 notebook 中 display(vis)
```

默认会写出 `*_mask_overlay.png`（与 mask 同目录）；如不落盘可传 `save=False`。

### 输出 meta 字段说明（用于复现实验/排障）

每次 `segment_image(...)` 会落盘一个 `*.json` 元信息文件（路径见 `result.meta_path`），便于复现与定位问题。核心字段如下：

| 字段 | 类型 | 含义 |
|------|------|------|
| `query_image_path` | str | 待分割原图绝对路径 |
| `retrieval.input_count` | int | 传入的 Top-k 条目数量（解析后） |
| `retrieval.usable_count` | int | 可用条目数量（mask 存在且可读） |
| `retrieval.usable[]` | list | 实际写入 memory bank 的条目列表（image/mask/distance） |
| `retrieval.skipped[]` | list | 被跳过的条目原因（缺字段、mask 不存在等） |
| `output_mask_path` | str | 输出 mask PNG 路径（单通道 8-bit，0/255） |
| `sam3_meta.sam3.version` | str | `"sam3"` 或 `"sam3.1"` |
| `sam3_meta.synthetic_video.frames[]` | list | “合成视频帧序列”的映射；最后一帧通常是 `query_image_path` |
| `sam3_meta.memory_locked` | bool | 是否锁定 memory bank（对应 `lock_memory`） |
| `sam3_meta.run_mem_encoder_on_target` | bool | 目标帧是否运行 memory encoder（锁定时为 `False`） |
| `sam3_meta.output_prob_thresh` | float | 输出二值化阈值（概率） |
| `sam3_meta.obj_id` | int | 当前输出对象 ID（默认 0） |

注意：`sam3_meta.synthetic_video.frame_dir` 指向临时目录（`ephemeral=true`），仅用于记录本次推理时的“物化帧”映射，后续可能已被系统清理，不应作为长期依赖路径。

#### retrieval_topk 支持的输入格式

`retrieval_topk` 可传入以下任意一种（可混用）：

- RAG tuple：`[(img_path, mask_path, distance), ...]`
- dict 列表：例如 `{"image_path": "...", "mask_path": "...", "distance": 1.2, "score": 0.9}`
- `seg_pipeline.protocol.RetrievalItem` 列表

缺失字段、mask 不存在等会被记录到 meta 的 `retrieval.skipped`，并按“跳过条目”降级处理。

## SAM3 memory bank 机制与锁定策略

SAM3 在单张图片模式下默认不会启用记忆注意力。`seg_pipeline` 的做法是：

1. 将 Top-k 样例图像 + 待测图像“物化”为一个仅包含 `0.jpg..N.jpg` 的目录，作为“合成视频帧序列”
2. 对 Top-k 帧逐帧 `add_new_mask` 写入专家掩码，调用 `propagate_in_video_preflight(run_mem_encoder=True)` 将这些条件帧编码进 memory bank
3. 对目标帧推理时，若 `lock_memory=True`，则以 `run_mem_encoder=False` 传播，确保 memory bank 在目标帧阶段保持只读

### 实现细节（sam3 vs sam3.1）

项目支持 `version="sam3"` 与 `version="sam3.1"` 两条上游 API 路径，二者初始化与缓存策略不同：

- `sam3.1`（multiplex）：使用上游的 multiplex video predictor / tracking 形态，按 `resource_path` 或 `video_path` 初始化状态并进行传播。
- `sam3`（dense video + tracker）：`add_new_mask` 会依赖每帧的 backbone 特征缓存（`cached_features`）。为保证“Top-k 作为条件帧写入”的流程可用，`seg_pipeline` 会在写入 mask 前预先对 `0..N` 所有帧缓存必要特征，并保留完整缓存快照，避免上游的逐帧缓存淘汰影响条件帧写入。

此外，为避免上游 `init_state(...)` 签名变化或位置参数误用导致 `video_height/video_width` 类型错误，`seg_pipeline` 会优先使用关键字参数（如 `video_path=`/`resource_path=`）初始化 tracker 状态。

## 依赖

- 根目录 `requirements.txt`：分割流水线 + SAM3 基础依赖 + 引用 `rag/requirements.txt`
- `rag/requirements.txt`：RAG 侧依赖（torch/transformers/faiss/Pillow 等）

说明：本仓库同时包含上游源码（`resource/sam3` 等），可按需要选择”源码引用”或”pip 安装”方式使用；当前 `seg_pipeline` 默认在导入失败时将 `resource/sam3` 临时加入 `sys.path` 以便直接使用本地源码。

### SAM3 权重下载

已登录 HuggingFace 且通过 Meta 授权后，使用项目提供的脚本下载：

```bash
python scripts/download_sam3_weights.py              # 默认 sam3.1 → ./sam3_weights/
python scripts/download_sam3_weights.py --version sam3  # sam3 版本
python scripts/download_sam3_weights.py -o /path/to/weights  # 自定义输出目录
```

脚本会下载对应版本的 `config.json` 和模型权重文件，并在末尾输出可直接填入 `Sam3BuildConfig(checkpoint_path=...)` 的路径。

## 测试（静态验证）

测试仅覆盖协议解析、RAG 适配、以及 memory bank 锁定参数传递（使用 mock/stub，避免加载模型权重与端到端推理）。

- `tests/test_protocol_parsing.py`
- `tests/test_rag_integration.py`
- `tests/test_memory_lock_passing.py`

为避免误收集 `resource/` 下的上游测试，已通过 `pytest.ini` 将收集范围限定到 `tests/`。

## 设计约束

- 输出掩码为 PNG 单通道 8-bit（0/255）
- Top-k 样例必须同时具备图像与对应专家 mask（mask 缺失条目会被跳过）
- 默认启用 memory bank 锁定（`lock_memory=True`），以保证检索记忆不被目标帧更新污染
