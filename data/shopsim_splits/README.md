# 冻结的 ShopSimulator 任务划分

本仓库不分发上游任务文本。设置 `SHOPMINI_UPSTREAM_DIR` 指向固定提交的上游 checkout 后，运行 `python scripts/freeze_shopsim_splits.py --write`，再运行 `--check`。生成的 `manifest.json` 记录源数据哈希和任务 ID；`grpo_train.jsonl` 是过滤掉 SFT 重叠任务后的 994 题训练池。两者仅保存在本地并被 Git 忽略。50 题开发集和 Final-200 保留在上游 checkout。
