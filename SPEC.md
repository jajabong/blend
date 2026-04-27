# Blend — SPEC.md 开发规范单文件

> **Version:** 2.0.0
> **Status:** DONE — 24/24 stories (v2.0.0)
> **Last Updated:** 2026-04-27

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

## 资源模型

| 模型 | 用途 | 层级 |
|------|------|------|
| Opus 4.7 | 复杂策略制定 | L2/L5(HIGH) |
| Sonnet 4.6 | 主开发执行 | L3(主) |
| Haiku 4.5 | 简单任务 | L3(轻量) |
| Minimax 2.7 | 压缩/简单任务 | L1/L4 |
| Gemini 3.1 | 批量推理 | L3(批量) |

---

## 故事清单

### 已完成（v1.3.0）

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

### 进行中（v2.0.0）

（无）

---

## 进度摘要

| 指标 | 值 |
|------|-----|
| 总故事数 | 24 |
| 已完成 | 24 |
| 进行中 | 0 |
| 待开始 | 0 |
| 完成率 | 100% |

---

## 技术约束

1. **配置文件**: models.yaml 管理模型映射，不允许硬编码
2. **测试覆盖**: 覆盖率 ≥ 80%（各模块独立阈值）
3. **L2 策略**: 必须调用真实 Opus，不得用规则占位符
4. **Fallback**: 所有 fallback 链必须可测试
5. **Prompt 模板**: 统一在 `blend/prompts/` 目录

---

## 开发命令

```bash
# Lint
cd /Users/dongshenglu/blend && ruff check .

# 类型检查
cd /Users/dongshenglu/blend && mypy .

# 测试 + 覆盖率
cd /Users/dongshenglu/blend && .venv/bin/python -m pytest tests/ -v --cov=blend --cov-report=term-missing

# Smoke test
cd /Users/dongshenglu/blend && blend status

# API 服务
cd /Users/dongshenglu/blend && blend serve
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

---

## 验证清单

- [ ] 版本号存在
- [ ] 触发词存在（4 个）
- [ ] 资源模型定义
- [ ] 故事清单完整（19 个）
- [ ] 开发命令正确
- [ ] 技术约束明确
