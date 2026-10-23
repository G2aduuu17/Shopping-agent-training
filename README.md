# ShopMini：单卡购物模型实验

本项目在单张 RTX 3090 Ti 上探索 Qwen3.5-2B 的多轮购物能力。原始工作区已完成 ShopSimulator 接入、Stage A/B/C 监督微调和冻结评测。**同一批 50 道开发题的 BF16 对照中，原始 Qwen 严格成功 0/50，Stage A SFT 为 24/50；有效奖励终局从 2/50 到 47/50。**这是该开发集上的明确改善，不是最终泛化成绩。

后续 Stage B/C 的监督验证损失各自在本阶段下降；按相同 FP32 开发集协议，严格成功分别为 Stage A **24/50**、Stage B **25/50**、Stage C **25/50**。Stage C 合并模型另在当前 Final-200 Clean 上得到 **123/200**，但该单卡 12K 推理协议与原仓库历史结果不同，不能直接比较。Stage B 的净增仅 1 题，Stage C 没有净增，不能写成持续提升。

本仓库从本地工作区整理出可复用的训练核心、测试、示例配置和部分 ShopSimulator 脚本，不包含完整 SFT 流水线、模型权重、checkpoint、上游任务数据、评测原始产物、虚拟环境或上游仓库。上述 SFT 数字来自原始工作区的冻结评测，不能仅靠本仓库重跑。`cpu-smoke` 只验证训练数学和保存/重载，不代表购物成功率。

当前在线训练已有采样、概率审计、更新和恢复的技术验收；原始工作区的连续训练状态在本次核对时记录 **34 次优化器更新**。尚未完成 Stage C 起点与所选 checkpoint 的同协议 50 题配对评测，因而**没有证据宣称在线训练提升购物成功率**。若传播训练结果，应另附可核查的结果报告，并区分开发集、最终集与训练技术验收。

核心包包含合成商品选择任务的 Qwen LoRA 训练、概率审计、断点恢复和独立重载；`scripts/` 提供 ShopSimulator 的冻结多轮评测、任务划分校验和 Stage C 的采样/更新入口。ShopSimulator 路径需要另行准备固定版本的上游环境、模型及已完成的 SFT 产物。

## 目录

```text
src/shopmini/          训练、奖励、rollout 和概率审计实现
tests/                 不依赖 GPU 的单元测试；部分上游测试有条件执行
configs/qwen_smoke.json 真实模型的示例配置
data/                  合成商品任务及 ShopSimulator 任务划分生成说明
scripts/               环境、冻结评测及 Stage C 采样/更新入口
docs/PUBLICATION_SCOPE.md 选入、排除范围与验证结果
```

## 最小验证

在本目录的 PowerShell 中，使用 Python 3.12/3.13 和与机器匹配的 PyTorch 安装本包：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e '.[dev]'
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m shopmini cpu-smoke --output outputs/cpu-smoke
```

查看 `outputs/cpu-smoke/summary.json`：应有 `checkpoint_reload: true`，且更新次数大于零。没有本地上游 checkout 时，上游相关测试会跳过。`doctor` 会检查实际 CUDA/BF16，GPU 机器上可另行运行：

```powershell
.venv\Scripts\python.exe -m shopmini doctor --output outputs/preflight.json
```

## 合成商品任务的真实模型测试

将已合法取得的本地 Qwen 模型放入忽略的 `models/qwen3.5-2b/`，并核对 `configs/qwen_smoke.json` 中的 `model_revision` 是否与模型版本一致。配置内相对路径以配置文件所在目录为基准。请在 GPU 空闲时先执行预检，再用新的输出目录训练：

```powershell
.venv\Scripts\python.exe -m shopmini train --config configs/qwen_smoke.json --dry-run
.venv\Scripts\python.exe -m shopmini train --config configs/qwen_smoke.json --output-dir outputs/my-qwen-run
```

这一路径是合成目录的单次商品选择任务，不能用它的结果代表 ShopSimulator 多轮购物能力。

## ShopSimulator 路径

将 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) checkout 到忽略的 `upstream/`，固定在 `dc6c9af8e48c8e6325101b39d1d20060f69e4221`。若 checkout 在别处，设置 `SHOPMINI_UPSTREAM_DIR` 为其绝对路径。程序会校验提交与已跟踪文件的清洁状态，不会将上游代码纳入本仓库。

```powershell
$env:SHOPMINI_UPSTREAM_DIR = (Resolve-Path ..\shopping-grpo-longhorizon).Path
.venv\Scripts\python.exe scripts/freeze_shopsim_splits.py --write
.venv\Scripts\python.exe scripts/freeze_shopsim_splits.py --check
.venv\Scripts\python.exe scripts/setup_shopsim.py --install
.venv\Scripts\python.exe scripts/serve_shopsim.py
```

ShopSimulator 服务使用独立的 Python 3.10 环境。`setup_shopsim.py` 需要 `uv` 和 `requirements-shopsim-windows.lock.txt`，请先在当前 `.venv` 安装 `uv`。服务启动后，在另一个终端进行接口或冻结模型检查，明确传入本地模型：

```powershell
.venv\Scripts\python.exe scripts/smoke_shopsim.py
.venv\Scripts\python.exe scripts/run_shopsim_baseline.py --model-path models/qwen3.5-2b --split validation --max-tasks 1 --output-dir outputs/frozen-baseline-001
```

Stage C GRPO 入口依次为 `prepare_stage_c_grpo.py`（校验并冻结已完成的 SFT 合并模型）、`stage_c_grpo_entry.py`（G=4 当前策略采样）和 `update_stage_c_grpo.py`（只消费有有效奖励差异且通过概率审计的组）。这些脚本要求显式传入 `--model`/`--sft-run`、`--start-manifest`、`--source` 与 `--output-dir` 等输入；本仓库没有附带前期 SFT 权重和历史运行目录。先查看各脚本 `--help`，再用自己的产物路径启动。

运行划分生成脚本后，ShopSimulator 的任务划分见本地 `data/shopsim_splits/manifest.json`。其中 994 个 GRPO 训练任务与 SFT 任务、50 个开发任务及 Final-200 分离。Final-200 只用于最终评测。

项目参考上游仓库的任务环境和数据；本仓库的实验代码与上游 checkout 分开。若要开放他人复用，请先明确自有代码的许可证，并检查模型与数据的分发权限。
