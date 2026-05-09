# MedSeg RAG System

基于 dinov3 和 FAISS 的医学图像检索系统。

## 功能特性

- 使用 **dinov3-vitl16** 模型提取图像 embedding（cls token + L2 归一化）
- 使用 **FAISS IndexFlatL2** 实现 L2 距离检索
- 支持图像与 mask 文件配对管理
- 提供增删检索库文件的简单接口

## 环境配置

```bash
conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt
```

## 目录结构

```
rag/
├── embedding/
│   └── dinov3_embedder.py    # dinov3 embedding 提取
├── index/
│   └── faiss_indexer.py       # FAISS 索引管理
├── storage/
│   └── file_manager.py        # 文件映射管理
├── config.py                  # 配置参数
├── rag_system.py              # RAG 系统主类
└── requirements.txt            # 依赖列表
```

## 快速使用

```python
from rag import RAGSystem

# 初始化
rag = RAGSystem(index_dir="./rag_index")

# 批量索引图像
rag.index_images(
    image_paths=["img1.png", "img2.png"],
    mask_paths=["mask1.png", "mask2.png"]
)

# 检索 top-5 结果
results = rag.search("query.png", k=5)
# 返回: [(图像路径, mask路径, 距离), ...]

# 添加单个图像
rag.add_image("new_image.png", "new_mask.png")

# 删除图像
rag.remove_image("img1.png")
```

## 接口说明

### RAGSystem

| 方法 | 说明 |
|------|------|
| `index_images(image_paths, mask_paths)` | 批量索引图像和 mask |
| `search(query_image, k)` | 检索 top-K 结果 |
| `add_image(image_path, mask_path)` | 添加单个图像 |
| `remove_image(image_path)` | 删除图像 |
| `get_indexed_count()` | 获取已索引数量 |
| `clear_index()` | 清空索引 |

### 配置参数

通过环境变量或 `config.py` 修改：

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| 模型名称 | - | `facebook/dinov3-vitl16` | dinov3 模型 |
| 索引目录 | `RAG_INDEX_DIR` | `./rag_index` | 索引存储路径 |
| 默认 top-k | - | `5` | 检索默认返回数量 |
| embedding 维度 | - | `1024` | dinov3-vitl16 输出维度 |

## 依赖

- torch
- torchvision
- transformers
- faiss-cpu
- Pillow
