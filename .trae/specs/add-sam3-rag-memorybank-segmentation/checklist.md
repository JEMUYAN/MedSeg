* [x] 分割输入支持两种模式：直接提供 retrieval\_topk；或提供 query\_image\_path+k 由 RAG 检索得到 top-k

* [x] Top-k 结果结构与 rag/RAGSystem.search 返回兼容：\[(img\_path, mask\_path, distance), ...]

* [x] SAM3 通过视频预测 API 启用记忆注意力：合成帧序列（Top-k 样例帧 + 目标帧），并把样例掩码作为 mask prompt 写入条件帧

* [x] Top-k 写入后锁定 memory bank：目标帧推理阶段禁用 memory encoder（语义等价 run\_mem\_encoder=False），避免污染记忆

* [x] mask 缺失/不可读时有明确的降级或错误策略，不导致整体崩溃

* [x] 输出掩码文件为 PNG 单通道 8-bit（0/255），并提供结构化元信息

* [x] requirements.txt 已提供（仅文件输出），且未包含任何环境配置/执行步骤

* [x] 单元测试覆盖输入校验与适配逻辑（使用 mock/stub，避免加载模型/运行端到端推理）

