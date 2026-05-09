# 基于 RAG Top-k 记忆库的 SAM3 分割 Spec

## Why
现有流程需要把待测图片与 RAG 检索到的带专家标注掩码的样例共同送入 SAM3 的记忆注意力机制，从而在缺少交互式提示（点/框）的情况下完成目标分割。

## What Changes
- 新增“分割流水线”能力：接收待测图片文件与 RAG Top-k 检索结果（图像路径 + 掩码路径 + 相似度），输出分割掩码图像文件。
- 新增 SAM3 适配层：基于 SAM3 的视频预测 API 构建 predictor，使用“合成视频帧序列”启用记忆注意力机制，并将 Top-k 样例写入 memory bank。
- 新增 memory bank 锁定机制：在 Top-k 写入完成后，分割待测帧时禁止继续写入 memory bank（保持只读），保证检索记忆不被后续步骤污染。
- 新增与 `rag/` 的可选集成：既支持直接传入 Top-k 结果，也支持输入 `k` 后由系统调用 RAG 检索得到 Top-k。
- 新增依赖清单输出：提供 `requirements.txt`（仅文件输出，不执行环境配置）。

## Impact
- Affected specs: “以检索样例作为记忆提示的单图分割（SAM3 memory attention）”
- Affected code:
  - RAG：复用 [RAGSystem.search](file:///Users/sagm/workspace/MedSeg/rag/rag_system.py#L85-L101) 的返回结构 `[(img_path, mask_path, distance), ...]`
  - SAM3：复用 [build_sam3_predictor](file:///Users/sagm/workspace/MedSeg/resource/sam3/sam3/model_builder.py#L1243-L1318) 以及 predictor 的 `handle_request/handle_stream_request` 分发
  - 新增：项目内新增一个“分割流水线模块”（文件路径待实现阶段确定）

## ADDED Requirements

### Requirement: 分割输入与输出
系统 SHALL 接收：
- `query_image_path`：待测图片文件路径
- `retrieval_topk`：Top-k 检索结果列表（每项至少包含 `image_path` 与 `mask_path`，并可包含 `score/distance`）

系统 SHALL 输出：
- `output_mask_path`：分割掩码文件（PNG，单通道 8-bit，0/255）

#### Scenario: Success case（直接提供 Top-k）
- **WHEN** 传入 `query_image_path` 与 `retrieval_topk`
- **THEN** 生成 `output_mask_path`
- **AND** 返回结构化元信息（例如：实际使用的 Top-k 条目数、每条目的路径与分数）

#### Scenario: Top-k 条目缺少掩码
- **WHEN** `retrieval_topk` 中存在 `mask_path` 为空或文件不存在的条目
- **THEN** 跳过该条目或降级处理（不得导致整体流程崩溃）

### Requirement: 可选的 RAG 检索集成
系统 SHALL 支持以 `query_image_path` 与 `k` 为输入，调用 RAG 系统获得 `retrieval_topk`，其结构与 [RAGSystem.search](file:///Users/sagm/workspace/MedSeg/rag/rag_system.py#L85-L101) 返回一致。

#### Scenario: Success case（系统内执行检索）
- **WHEN** 传入 `query_image_path` 与 `k`
- **THEN** 调用 RAG 检索获得 `retrieval_topk`
- **AND** 继续执行分割并产生 `output_mask_path`

### Requirement: SAM3 记忆库写入与分割推理
系统 SHALL 使用 SAM3 的“视频预测”能力启用记忆注意力机制，并按如下逻辑工作：
- 将 `retrieval_topk` 的样例图像作为“记忆帧/条件帧（conditioning frames）”
- 将 `query_image_path` 作为“待分割帧（target frame）”
- 对每个记忆帧，将对应专家掩码以“mask prompt”的形式加入该帧，使 SAM3 运行记忆编码器并写入 memory bank
- 对待分割帧，执行传播/推理得到该帧的输出掩码（作为最终结果）

#### Scenario: 合成视频帧序列启用记忆注意力
- **WHEN** `retrieval_topk` 有 N 条可用样例
- **THEN** 系统将构造一个包含 `N + 1` 帧的“合成视频”输入（前 N 帧为检索样例，最后 1 帧为待测图片）
- **AND** 通过 predictor 的 request/stream API 完成 session 初始化、加入条件帧提示、传播到目标帧并获取输出

### Requirement: 锁定 memory bank（只读）
系统 SHALL 在完成 Top-k 样例写入 memory bank 后“锁定”记忆库，使后续对待分割帧的推理过程中不再写入/更新 memory bank。

锁定的最小行为定义：
- 对目标帧推理/传播时，必须禁用 memory encoder 的运行（等价于以 `run_mem_encoder=False` 的语义执行目标帧推理），从而避免将目标帧或中间结果写回 memory bank。

#### Scenario: Top-k 写入后锁定
- **WHEN** Top-k 条目写入完成
- **THEN** 目标帧推理阶段 memory bank 保持只读（Top-k 记忆不被覆盖/稀释）

### Requirement: 依赖清单输出（不配置环境）
系统 SHALL 在仓库中提供 `requirements.txt` 以描述该能力所需依赖版本范围。

约束：
- 不得在开发过程中执行环境配置（例如 conda 创建环境/安装依赖）
- 不得在开发过程中运行项目代码（包括启动服务或执行端到端推理）

## MODIFIED Requirements
（无）

## REMOVED Requirements
（无）

