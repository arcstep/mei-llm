# SFT alignment stages

Bootstrap 与 adaptive 是两个可恢复的 pipeline，不是由 exposure 标签选择的
脚本版本。pipeline ID、recipe SHA、Base/QAT/SFT/Eval 输入和所有 step/limit
参数都由 binding 冻结。Bootstrap 的正式输出边界是 `tool_index_v4`；
adaptive 才训练 MW/confidence/narration。SFT、linguistic 和 eval 的具体
release ID 由 binding 指定，schema 合同仍由代码校验。

阶段中允许使用 dev/validation 做训练控制，但 locked test 不得用于调阈值。
训练阶段结束后，已完成 receipt 可由后续阶段复用；input fingerprint 改变时必须
创建新 run，禁止在旧 run 上混入新语料或新架构。
