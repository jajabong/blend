# ARCHITECTURE.md — Blend 架构决策记录

> 本文件记录 blend 项目的重要架构决策，供 AI 在首次运行时读取。
> 主要回答"为什么这样做"而非"做什么"。

## 1. 独立系统决策

**问：** blend 和 openblend 是什么关系？

**答：** 两个独立系统，解决不同问题：
- **openblend** → "用哪个模型最好？"（模型选择引擎）
- **blend** → "如何用最少 Token 达到最好质量？"（Token 优化引擎）

**原因：** 从第一性原理分析，两个问题的优化目标完全不同。强行合并会导致系统臃肿且两者都做不好。

## 2. 五层固定架构决策

**问：** 为什么用固定五层而非 openblend 的动态路由？

**答：** 商用场景需要可预测性：
- 固定架构 → 成本可预测 → 可向用户承诺 SLA
- 动态路由 → 效果更好但成本不可控 → 无法做商用定价

**原因：** blend 的核心价值是"固定成本 + 可承诺质量"，这要求架构稳定。

## 3. Minimax 作为 L1+L4 决策

**问：** 为什么 L1 和 L4 都用 Minimax？

**答：** 成本结构最优：
- L1 需要"理解 + 压缩" → Minimax 语义压缩能力强，成本极低
- L4 需要"压缩 >500T 输出" → 再次用 Minimax 成本最低

**原因：** Minimax 20 CNY/月无限 Token，是 blend 成本控制的核心杠杆。

## 4. L2 输出 ≤300T 决策

**问：** 为什么 L2 Opus 输出必须 ≤300T？

**答：** 成本控制关键点：
- Opus 688 CNY/月套餐，月度清零
- L2 只做策略，不做执行 → 不应消耗大量 Token
- 300T 是 L2 纯策略输出的合理上限

**原因：** L2 如果 >300T，策略层会成为成本黑洞。

## 5. Gemini 批量规则决策

**问：** 为什么强制 Gemini ≥50% 上限才调用？

**答：** Gemini 按次计费，长期有效：
- <50% 上限 → 资源浪费，等归集更合算
- ≥50% 上限 → 充分利用上下文窗口

**原因：** Gemini 的成本优势和批量调用是天作之合。

## 6. 复用 openblend 资产决策

**问：** blend 可以复用 openblend 哪些代码？

**答：** 可复用（工程资产）：
- `openblend/blend/core/types.py` → Tier/Strategy/BlendMode
- `openblend/blend/config.py` → YAML 配置 + dotenv
- `openblend/blend/providers/unified.py` → HTTP/CLI/MCP 统一接口

**不可复用（架构设计）：**
- ELO 系统 / Arena 基准 / Swarm Debate / 50+ 模型池
- 原因：blend 是固定五层，不需要自适应选择

## 7. 不复用 intent/analyzer_v2.py 决策

**问：** 为什么不用 openblend 的 intent/analyzer_v2.py？

**答：** 12 维分类对 blend 来说过度设计：
- openblend 需要精细化模型选择 → 需要 12 维
- blend 只需要 1-10 复杂度评分 → 简化更高效

**学习思想：** 从 12 维提取"复杂度评分"思想，简化为 5 维（步骤/领域/输出/创意/风险）。

## 8. Haiku 专属 L3 决策

**问：** 为什么 Haiku 有专属层级而不只是 Sonnet 备选？

**答：** Haiku 成本优势显著：
- Haiku 4.5 = 90% Sonnet 能力，1/3 成本
- 复杂度 3-5 分任务不需要 Sonnet
- Haiku 专属 → 明确告诉 AI "用 Haiku，别浪费 Sonnet 额度"

**原因：** 成本精算决定细节，Haiku 是 blend 成本控制的关键角色。

## 9. L2 Opus 集成决策

**问：** L2 为什么直接调用 Opus 而非规则占位符？

**答：** 复杂任务需要真正的策略推理：
- 规则占位符无法生成高质量执行计划
- Opus 的深度推理能力最适合策略生成
- 通过 BaosiProvider 调用 claude-opus-4-7，JSON 解析策略输出

**实现：** `strategy.py` 的 `_call_opus()` 方法调用 BaosiProvider，解析 JSON 响应。复杂度 ≥6 时触发（与 `scorer._determine_tier()` HIGH 阈值一致），失败时回退到规则生成。

**Fallback 链：** Opus 失败 → `_generate_plan()` / `_generate_redlines()` / `_identify_boundary_cases()`

## 11. Prompt 模板统一管理决策

**问：** 为什么 Prompt 模板要独立到 `blend/prompts/` 目录？

**答：** 分离关注点，便于维护：
- L2 策略注入模板 (`strategy.py`): 包含执行计划文本
- 避免硬编码字符串在业务逻辑中
- 测试时可独立验证模板渲染

**文件：** `blend/prompts/strategy.py` 包含 `L2_STRATEGY_SYSTEM_TEMPLATE`

---

## 10. 三层 Executor 分流决策（v1.7.0）

**问：** 为什么从 Minimax 直通升级到三层 Haiku 分流？

**答：** ROI 分析证明 Haiku 全面优于 Minimax：
- Haiku 4.5 = 90% Sonnet 能力，1/3 成本
- 成本对比：Tier1 任务 Haiku 4.5 vs Minimax M2.7 → 质量提升远超成本增量
- Phase 1 ROI 分析：L1 移除后，Tier1 任务不需要 Minimax 的语义压缩优势

**分流架构：**
| Tier | Complexity | Primary | Fallback | 理由 |
|------|-----------|---------|----------|------|
| Tier 1 | 1-2 | Haiku | Minimax | Haiku 质量远超成本增量 |
| Tier 2 | 3-5 | Haiku/Sonnet | Haiku→Minimax | Sonnet 仅当 budget>100T |
| Tier 3 | 6-10 | Sonnet | Haiku→Minimax | 高复杂度需要 Sonnet |

**Haiku 阈值统一：** Tier1/2/3 的 Haiku fallback 门槛统一为 ≥50T（原来 Tier2/3 用 >100，Tier1 用 >50）。

## 11. Opus Advisor Loop（v1.8.0）

**问：** `_determine_model_hint` 为什么用硬编码规则？

**答：** 旧实现：复杂度 ≥9 → "Opus"，否则 → "Sonnet"。这是静态规则，忽略了任务实际特征。

**改进：** Opus 生成 plan 时同时输出 `model_recommendation`：
- 扩展 `OPUS_SYSTEM_PROMPT` 请求 `model_recommendation` 字段
- `_call_opus()` 返回 4 值：plan、redlines、boundary_cases、model_recommendation
- `generate()` 优先使用 Opus 的推荐，空白时回退规则

**Fallback 链：** Opus 返回 model_recommendation → 使用；空字符串 → `_determine_model_hint(complexity)`；API 失败 → 规则生成

## 12. L5 Gemini 质检决策（v1.9.0）

**问：** L5 质量门为什么用规则模式而非语义推理？

**答：** 旧实现：纯字符串模式匹配（taboo/secrets/vuln patterns）。对 HIGH 复杂度任务质量评估不足。

**改进：** HIGH 复杂度任务调用 Gemini 语义评估：
- 新增 `gemini_evaluate()` 方法：relevance / accuracy / completeness 三维评分
- `verify()` 中 Gate 9：`quality_level == "HIGH"` 时触发 Gemini 评估
- `GEMINI_QUALITY` gate 合并到整体 gates dict
- API 失败 → safe default (passed=True)，不阻断流程

**Gate 扩展：** 8 → 9（base gates）

## 13. v2.0 代码清理（v2.0.0）

**清理项：**
- ruff check: 0 errors / 0 warnings（blend/ + tests/）
- 未使用 import 清理（test_l2.py, test_phase3, test_phase4）
- 变量命名规范（MockProvider → mock_provider）
- 测试文件架构完整性（test_l5.py gate count 11→12）

**质量指标：**
- 测试：496 passed（mock only），覆盖率 90%
- lint：All checks passed

**最后更新：** 2026-04-27（v2.0.0 — 代码清理完成）
