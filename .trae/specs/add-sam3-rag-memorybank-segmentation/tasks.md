# Tasks
- [x] Task 1: 定义分割能力的输入/输出协议
  - [x] 明确 `retrieval_topk` 的允许字段（image_path、mask_path、score/distance 等）与校验规则
  - [x] 明确输出掩码格式（PNG、单通道 8-bit、0/255）与元信息结构

- [x] Task 2: 实现 SAM3 适配层（视频预测 + 记忆写入）
  - [x] 基于 [build_sam3_predictor](file:///Users/sagm/workspace/MedSeg/resource/sam3/sam3/model_builder.py#L1243-L1318) 构建 predictor（优先尝试 transformers；不满足则使用本地源码 + 权重路径配置）
  - [x] 构造“合成视频帧序列”（Top-k 样例帧 + 目标帧）并建立 session
  - [x] 将每个 Top-k 样例的专家掩码作为该帧的 mask prompt 写入 conditioning outputs（触发 memory encoder）

- [x] Task 3: 实现 memory bank 锁定逻辑
  - [x] 在 Top-k 写入后，对目标帧推理阶段强制禁用 memory encoder（语义等价 `run_mem_encoder=False`）
  - [x] 保证锁定后不会对既有 `cond_frame_outputs` 做破坏性修改

- [x] Task 4: 实现与 RAG 系统的对接（可选）
  - [x] 支持直接消费 [RAGSystem.search](file:///Users/sagm/workspace/MedSeg/rag/rag_system.py#L85-L101) 返回结构 `[(img_path, mask_path, distance), ...]`
  - [x] 支持在仅提供 `query_image_path` 与 `k` 时调用 RAG 检索得到 `retrieval_topk`
  - [x] 对缺失/不可读的 mask 条目做降级处理（跳过或报可解释错误）

- [x] Task 5: 实现掩码落盘与结果产物
  - [x] 将目标帧输出掩码保存为 PNG（单通道 8-bit，0/255）
  - [x] 输出结构化元信息（例如 JSON：使用的 top-k、分数、实际处理条数、输出路径）

- [x] Task 6: 输出 requirements.txt（仅文件）
  - [x] 汇总本能力所需依赖（SAM3、RAG、图像 I/O、基础数值库等）
  - [x] 不执行任何环境配置动作

- [x] Task 7: 增加可静态验证的测试与示例（不运行）
  - [x] 为输入校验、Top-k 解析、RAG 结果适配编写单元测试（允许使用 mock/stub 避免加载模型权重）
  - [x] 增加最小“伪 predictor”测试替身，验证 memory bank 锁定参数/分支被正确传递

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 1
- Task 5 depends on Task 2 and Task 3
- Task 7 depends on Task 1 and Task 5
