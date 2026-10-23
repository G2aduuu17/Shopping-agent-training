# 发布准备范围

这份副本与原始 `mini/` 和独立上游 checkout 分开；整理过程没有修改原始源码或正在写入的训练结果。已在本地 Git `main` 分支提交，尚未连接远程仓库或上传。

## 选入

- `src/shopmini/`：单卡训练、采样、目标函数、checkpoint、ShopSimulator 客户端与多轮 rollout。
- `tests/`：与选入代码直接相关的单元测试。上游依赖测试在没有 pinned checkout 时跳过；依赖旧实验产物的两个 Stage C 测试未选入。
- `configs/qwen_smoke.json`、`data/catalog_tasks.jsonl`：可解释的合成目录实验配置与小任务集。
- `data/shopsim_splits/README.md` 与 `freeze_shopsim_splits.py`：从用户自己的固定版本上游 checkout 在本地生成任务划分；派生文件被 Git 忽略，不随公开副本分发。
- `scripts/`：环境设置和接口检查、冻结 Qwen 评测、概率信号检查、任务划分校验、Stage C 起点校验与 G=4 采样/更新。所有需要模型或前期产物的公开入口要求显式路径。
- `requirements-shopsim-windows.lock.txt`：Windows ShopSimulator CPU 服务的固定依赖快照。

## 未选入

- `mini/outputs/`、`models/`、虚拟环境、缓存、运行时索引、日志和下载的驱动：体积大，可能含机器状态或权重，不属于代码仓库。
- 本地 `upstream/` checkout：它是独立的外部项目，需按指定提交另行取得。
- 上游派生的 `data/shopsim_splits/{manifest.json,grpo_train.jsonl}`：保留生成脚本，不在公开仓库分发任务文本。
- 带日期的实验编排、预算恢复、消融和一次性比较脚本，以及依赖旧实验目录的正式协调器：当前副本保留基础采样/更新入口，后续若要公开完整自动编排，需先去掉历史路径并补齐独立验证。
- SFT 训练流水线和模型合并脚本：此版只支持消费已完成的 Stage C SFT 合并产物；不能声称从零复现完整 SFT→GRPO 结果。

## 本地验收

- 不接上游：`37 passed, 6 skipped`；跳过项需要 pinned 上游 checkout。
- 指定已固定版本的本机上游：`43 passed`。
- `shopmini cpu-smoke`：14 次更新，`checkpoint_reload=true`。这是 CPU 数学检查，不是 LLM 购物评测。
- 在本机固定上游 checkout 上生成并校验任务划分：任务数 1073/119（SFT 训练/验证）、994/50（GRPO 训练/开发）、200（最终集）；生成文件不纳入 Git。

Git `add --dry-run` 预览为 46 个代码、配置和文档文件；权重、输出、缓存以及本地生成的上游任务划分均被忽略。公开发布前仍需决定自有代码的许可证、在干净环境中安装验证，并决定是否补齐从零 SFT 与连续 Stage C 自动训练入口。此次整理不等于训练已完成，也不提供购物成功率提升的证据。
