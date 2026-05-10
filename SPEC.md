# Blend — SPEC.md 开发规范单文件

> **Version:** 2.1.0
> **Status:** ACTIVE — Advisor-Judge Architecture + Scheduler Modules + Advisor Optimization (verify_plan + self-verification)
> **Last Updated:** 2026-05-08

---

## 角色

你是 Blend 开发工程师，遵循 PromptForge 规范执行全自动开发。

**核心能力：**
- 生成/更新 SPEC.md 规范文档
- TDD 开发：先写测试 → 实现 → 重构
- 质量门验证：ruff → mypy → pytest → coverage ≥ 80%
- 完全无人化，不询问，直接执行

**执行原则：**
- 失败 2 轮无改进 → 停止，报告状态
- 覆盖率 ≥ 80% 才算完成
- 全部故事 DONE → smoke test → 交付

---

## Advisor-Judge 三层架构

### 资源模型（Mac mini M4 16G/256G）

| 模型 | 用途 | 配额规则 | 策略 |
|------|------|----------|------|
| MiniMax M2.7 | 轻量执行层 | 4000次/5小时窗口 | fill_token (每次用满4096T) |
| Gemini 3 | 弹性催化剂 | 15000次/无时间限制 | batch_then_catalyst (单次吃满64K T) |
| Claude | 顾问+终审 | 授信额度 | advisor_judge_only (输出限制1024T) |

### 三层调度规则

| 复杂度 | 流程 | 模型选择 |
|--------|------|----------|
| LOW (1-2) | MiniMax直连 | `minimax` 执行 |
| MEDIUM (3-5) | Gemini初稿 → Claude评审 | `gemini_pro` → `claude_sonnet` |
| HIGH (6+) | Gemini初稿 → Claude(Advisor) → Gemini修正 → Claude(Judge)终审 | `gemini_pro_ultra` → `claude_opus` |

### 模型角色

- **EXECUTOR (Minimax)**: 基础执行层，5h窗口4000次，每次用足Token
- **CATALYST (Gemini)**: 弹性催化剂，15000次，单次吃满64K Token
- **ADVISOR_JUDGE (Claude)**: 顶级顾问+终审裁判，仅做催化剂，输出限制1024 Token

---

## 故事清单

### 已完成（v2.2.0）

- [x] **Story 1:** 项目脚手架 @ 2026-04-24
- [x] **Story 2:** L1 入口层 @ 2026-04-24
- [x] **Story 3:** L2 策略层 @ 2026-04-24
- [x] **Story 4:** L3 执行层 @ 2026-04-24
- [x] **Story 5:** L4 压缩层 @ 2026-04-24
- [x] **Story 6:** L5 终审层 @ 2026-04-24
- [x] **Story 7:** Enforcement 机制 @ 2026-04-24
- [x] **Story 8:** 资源模型 @ 2026-04-24
- [x] **Story 9:** API 服务 @ 2026-04-24
- [x] **Story 10:** 集成测试 + Smoke Test @ 2026-04-24
- [x] **Story 11:** 生成 SPEC.md（PromptForge 规范）@ 2026-04-25
- [x] **Story 12:** L2 接入真实 Opus 调用 @ 2026-04-25
- [x] **Story 13:** Provider 集成测试补全 @ 2026-04-25
- [x] **Story 14:** Fallback 路径测试覆盖 @ 2026-04-25
- [x] **Story 15:** 模型选择配置化（YAML）@ 2026-04-25
- [x] **Story 16:** Prompt 模板统一管理 @ 2026-04-25
- [x] **Story 17:** Tool Calling / Multimodal / JSON Mode 参数透传 @ 2026-04-26
- [x] **Story 18:** Full Agentic Tool Execution Loop (v1.4) @ 2026-04-26
- [x] **Story 19:** Agent Mode + MCP Tools + Context Budget (v1.5) @ 2026-04-26
- [x] **Story 20:** Forward OpenAI Params (top_p, presence_penalty, frequency_penalty, stop) @ 2026-04-26
- [x] **Story 21:** Anthropic Messages API Endpoint (Claude Code compatible) @ 2026-04-26
- [x] **Story 22:** Phase 2: Three-tier Executor (Tier1→Haiku, Tier2→Haiku/Sonnet, Tier3→Sonnet) @ 2026-04-27
- [x] **Story 23:** Phase 3: Opus Advisor Loop — model_hint driven by Opus reasoning @ 2026-04-27
- [x] **Story 24:** Phase 4: L5 Gemini Quality Gate (HIGH complexity) @ 2026-04-27
- [x] **Story 25:** 版本号统一（__init__.py + api.py → 2.0.0）@ 2026-04-27
- [x] **Story 26:** 阈值配置统一（executor._get_budget → ResourceModel）@ 2026-04-27
- [x] **Story 27:** mypy 类型修复（verifier + circuit_breaker）@ 2026-04-27
- [x] **Story 28:** MiniMaxDispatcher 滑动窗口限速（5h/4000次）@ 2026-05-07
- [x] **Story 29:** GeminiBatchQueue 批量队列（15000次）@ 2026-05-07
- [x] **Story 30:** QuotaMonitor 实时告警（WARNING/CRITICAL/EXHAUSTED）@ 2026-05-07
- [x] **Story 31:** MonthlyReseter 月度自动重置 @ 2026-05-07
- [x] **Story 32:** TokenFiller Token填充优化 @ 2026-05-07
- [x] **Story 33:** Executor调度器集成（dispatcher/alert/filler → Executor）@ 2026-05-07
- [x] **Story 34:** Python 3.9 兼容修复（__future__ annotations、f-string反斜杠、Pydantic类型注解）@ 2026-05-08
- [x] **Story 35:** Advisor优化 - verify_plan预检、self-verification、HIGH流程翻转 @ 2026-05-08

### 进行中（v2.2.0）

（无）

---

## Scheduler 模块

### MiniMaxDispatcher

```python
class MiniMaxDispatcher:
    WINDOW_HOURS = 5
    MAX_CALLS = 4000
    MIN_INTERVAL_SECONDS = WINDOW_SECONDS / MAX_CALLS  # ~4.5秒

    def can_dispatch(self) -> bool:
        # 检查窗口额度 + 匀速间隔

    def dispatch(self, task: str) -> str | None:
        # 记录调用并返回任务

    def reset_window(self) -> None:
        # 重置5小时窗口
```

### GeminiBatchQueue

```python
class GeminiBatchQueue:
    MAX_CALLS = 15000
    TOKEN_TARGET = 64000

    def add(self, task_content: str) -> BatchResult | None:
        # 添加任务，批量或超时触发flush

    def can_add(self) -> bool:
        # 检查剩余额度

    def record_call(self) -> None:
        # 记录一次调用
```

### QuotaMonitor

```python
class AlertLevel(Enum):
    WARNING = "warning"    # <20% remaining
    CRITICAL = "critical"  # <5% remaining
    EXHAUSTED = "exhausted" # 0 remaining

class QuotaMonitor:
    WARNING_THRESHOLD = 20.0
    CRITICAL_THRESHOLD = 5.0

    def update_quota(self, model: str, remaining: int, max_calls: int) -> None:
        # 自动触发告警回调
```

### MonthlyReseter

```python
class MonthlyReseter:
    def needs_monthly_reset(self) -> bool:
        # 每月1日检查

    def force_reset(self) -> None:
        # 强制重置

    def reset_minimax_dispatcher(self, dispatcher: MiniMaxDispatcher) -> QuotaResetInfo:
        # 重置5小时窗口
```

---

## 进度摘要

| 指标 | 值 |
|------|-----|
| 总故事数 | 35 |
| 已完成 | 35 |
| 进行中 | 0 |
| 完成率 | 100% |

---

## 技术约束

1. **配置文件**: models.yaml 管理模型映射，不允许硬编码
2. **测试覆盖**: 覆盖率 ≥ 80%（各模块独立阈值）
3. **L2 策略**: 必须调用真实 Opus，不得用规则占位符
4. **Fallback**: 所有 fallback 链必须可测试
5. **Prompt 模板**: 统一在 `blend/prompts/` 目录
6. **Advisor-Judge**: Claude仅做催化剂，不做基础生成

---

## 开发命令

```bash
# Lint
cd /Users/dongshenglu/blend && ruff check .

# 类型检查
cd /Users/dongshenglu/blend && mypy blend/ --ignore-missing-imports

# 测试 + 覆盖率
cd /Users/dongshenglu/blend && python3 -m pytest tests/ -v --cov=blend --cov-report=term-missing

# Smoke test
cd /Users/dongshenglu/blend && curl http://127.0.0.1:8000/health

# API 服务
cd /Users/dongshenglu/blend && python3 -m uvicorn blend.api:app --host 127.0.0.1 --port 8000
```

---

## 触发词

| 触发词 | 行为 |
|--------|------|
| `blend 开发` | 开始开发循环 |
| `blend 验证` | 运行质量门验证 |
| `blend 状态` | 报告故事进度 |
| `blend SPEC` | 生成/更新 SPEC.md |

---

## 铁律

1. **完全无人化**：不询问，直接执行
2. **TDD**：先写测试（RED）→ 实现（GREEN）→ 重构（IMPROVE）
3. **覆盖率**: 各模块独立 ≥ 80%，总覆盖率 ≥ 80%
4. **L2 真实调用**: 禁止规则占位符，必须调用真实 API
5. **配置化**: 模型映射必须从 YAML 加载，禁止硬编码
6. **Advisor-Judge**: Claude仅做催化剂，Token消耗最小化

---

## 验证清单

- [x] 版本号存在 (2.2.0)
- [x] 触发词存在（4 个）
- [x] 资源模型定义 (Advisor-Judge)
- [x] 故事清单完整（33 个）
- [x] 开发命令正确
- [x] 技术约束明确
- [x] Scheduler模块文档