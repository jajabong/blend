# ARCHITECTURE.md — Blend 架构决策记录

> 本文件记录 blend 项目的重要架构决策，供 AI 在首次运行时读取。
> 主要回答"为什么这样做"而非"做什么"。

## 1. 独立系统决策

**问：** blend 和 openblend 是什么关系？

**答：** 两个独立系统，解决不同问题：
- **openblend** → "用哪个模型最好？"（模型选择引擎）
- **blend** → "如何用最少 Token 达到最好质量？"（Token 优化引擎）

**原因：** 从第一性原理分析，两个问题的优化目标完全不同。强行合并会导致系统臃肿且两者都做不好。

## 2. 四层固定架构决策 (v2.1.0 优化)

**问：** 为什么用固定四层而非 openblend 的动态路由？

**答：** 商用场景需要可预测性：
- 固定架构 → 成本可预测 → 可向用户承诺 SLA
- 改进：L2 触发门槛下调至 5+（原 6+），让更多中高难度任务获得 Opus 策略指导。
- 改进：新增"Scale 维度"评分，识别亿级/全球等超大规模需求并自动升阶。

**原因：** blend 的核心价值是"固定成本 + 可承诺质量"。通过下调策略层门槛和增加规模感知，在极小成本增量下换取了大幅质量提升。

## 3. Minimax 作为 L1+L4 决策

**问：** 为什么 L1 和 L4 都用 Minimax？

**答：** 成本结构最优：
- L1 需要"理解 + 压缩" → Minimax 语义压缩能力强，成本极低
- L4 需要"压缩 >500T 输出" → 再次用 Minimax 成本最低

**原因：** Minimax 20 CNY/月无限 Token，是 blend 成本控制的核心杠杆。

## 8. Haiku 专属 L3 & 代码质量底线决策 (v2.1.0)

**问：** 为什么 Haiku 有专属层级而不只是 Sonnet 备选？针对代码任务有何特殊逻辑？

**答：** 成本优势显著：
- Haiku 4.5 = 90% Sonnet 能力，1/3 成本。
- **新决策 (v2.1.0):** 代码任务 (TaskType.CODE) 实施"质量底线"。除非预算耗尽，否则代码任务严禁降级到 Minimax。
- 即使复杂度 ≤ 2，代码任务也优先路由至 Haiku 以确保逻辑严密。

**原因：** 成本精算决定细节，Haiku 是 blend 成本控制的关键角色。Minimax 在处理代码逻辑时易产生幻觉，确保代码任务的稳定性是商用 API 的生命线。

## 12. L5 质检与硬拦截决策 (v2.1.0 强化)

**问：** L5 质量门为什么执行硬拦截？

**答：** 旧实现：仅作为元数据标记，不阻断输出。

**改进：** 实施 Hard Deny 机制：
- 任何 Gate 失败（特别是 P0 漏洞、Taboo 内容、密钥泄露）都会**强制替换**输出为 `[REJECTED]` 错误。
- P0 漏洞检测升级为正则表达式，支持 `eval \s* (` 等变体识别。
- HIGH 复杂度任务继续强制 Gemini 语义质检，若质检不通过则拦截。

**原因：** 安全和合规是商用底线，宁可拒答，不可错答。

## 13. 不可摧毁的心脏架构 (v2.2.0 突破)

**问：** 面对 API 供应商（如 Baosi/Lemon）长达几天的宕机，系统如何生存？

**答：** 引入"自主生存协议"：
- **持久化健康记忆：** 熔断状态存入 `.blend_health.json`。重启后瞬间避障，无需再次尝试已确认死亡的 Provider。
- **并行赛跑 (Racing Fallback)：** 同时启动 Primary 和 Fallback。若 Primary 3秒无响应，Fallback 立即进场竞争。谁快用谁，用户感知延迟降至秒级。
- **指数退避：** 对故障 Provider 实施动态锁定（60s -> 1h -> 24h），区分技术抖动与行政欠费。

---

## 14. "榨干" Minimax 与草稿-精修架构 (v2.3.0)

**问：** 既然 Minimax 几乎免费，如何最大化其价值？

**答：** 实施 "Draft-Refine" 模式：
- **L1 预处理：** Minimax 不再只评分，而是为所有复杂度 ≥ 4 的任务先写一份"免费草稿 (DRAFT)"。
- **L3 精修：** 精英模型 (Sonnet/Gemini) 拿到草稿后进行"Review and Finalize"。
- **收益：** 减少了昂贵模型的思考发散性，大幅降低了按量计费 (Baosi) 的 Token 消耗，同时通过"免费脑力"提升了长文稳定性。

---

## 15. 自愈重试与"慈悲门禁" (Mercy Gate)

**问：** 质检失败只能报错吗？用户拿不到结果怎么办？

**答：** 建立"以用户为中心"的最后防御：
- **带反馈重试：** 质检失败时，将具体原因（如：缺少 input 校验）喂回模型原地重写。
- **慈悲交付 (Mercy Gate)：** 若重试后仍有轻微瑕疵，只要不涉及 P0 安全漏洞，系统会带上 `# --- QUALITY WARNING ---` 强行交付结果。
- **原则：** 拒绝垃圾代码，但绝不让用户空手而归。

---

## 16. 统筹精算路由决策

**问：** 面对按量计费 (Baosi) 和按次计费 (Lemon)，如何选？

**答：** 流量分配逻辑：
- **短平快任务：** 路由至 Baosi (Sonnet)。由于 Token 少，按量计费比按次更省。
- **长篇大论任务：** 路由至 Lemon (Gemini)。按次计费（10.5 额度）输出万字长文的 ROI 最高。
- **所有评分/草稿：** 强制 Minimax。

---

## 17. Claude Code 商用级接入 (v2.3.1)

**问：** Blend 如何实现作为 Claude Code 后端的商用级兼容性？

**答：** 通过"伪装握手 + 协议兼容"实现无缝接入：

### 17.1 模型名称伪装

Claude Code 启动时验证 `/v1/models` 端点。Blend 返回：
```json
{"id": "claude-3-5-sonnet-20241022", "object": "model", ...}
```
实际执行时自动映射到真实策略（Haiku/Sonnet/Gemini），用户无感知。

### 17.2 SSE 流式协议兼容

Blend 的 `/v1/messages` 端点输出符合 Anthropic SDK 规范的 SSE 事件：
```
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}

event: message_stop
data: {"type":"message_stop","message":{"id":"...","type":"message",...}}
```

### 17.3 Claude Code 工具调用支持

Blend 输出的 `[TOOL_CALL]` 块格式：
```json
{"tool": "read_file", "args": {"path": "blend/core/budget.py"}}
```
Claude Code 收到后自动执行并反馈结果，实现自主文件操作循环。

### 17.4 接入配置

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000/v1
export ANTHROPIC_API_KEY=blend-commercial-token
```

---

## 18. OpenCode + Blend 集成 (v2.3.1)

**问：** OpenCode 如何正确接入 Blend？

**答：** OpenCode 1.14.20 已验证支持 Blend：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| OpenCode config | `primary: "blend/blend"` | 已配置 |
| `ANTHROPIC_BASE_URL` | `http://localhost:8000/v1` | 核心修复 |
| `ANTHROPIC_API_KEY` | `blend-commercial-token` | 非空即可 |
| `BLEND_LOG_LEVEL` | `DEBUG` | 商用初期监控 |

### 18.1 OpenCode TUI 使用

```bash
# 终端 1: 启动 Blend
python3 -m uvicorn blend.api:app --host 0.0.0.0 --port 8000

# 终端 2: 启动 OpenCode
source ~/.zshrc  # 加载环境变量
opencode
```

### 18.2 CLI 快捷命令

```bash
# ~/.zshrc 中已配置
alias claude-mini='...'    # MiniMax M2.7
alias claude-baosi='...'    # Baosi Claude 3.5
alias claude-blend='...'    # Blend 商用优化
alias claude="claude-mini"  # 默认指向 MiniMax
```

### 18.3 性能基准

| 指标 | 实测值 |
|------|--------|
| 握手延迟 | < 100ms |
| 低复杂度任务 (L1>L3>L5) | ~3s |
| 高复杂度任务 (含 L2 策略) | ~60s |
| 流式输出稳定性 | 60s+不断连 |
| 代码生成质量 | 18,000+ 字符/次 |

---

**结论：** Blend 已完全具备作为 Claude Code / OpenCode 后端的商用级兼容性。通过 Minimax 草稿 + 精英模型精修的架构，成本比直连 Baosi 低 70%+，同时保持 Claude Code 级别的工程质量。

---

## 17. Claude Code 商用接入与协议伪装决策 (v2.4.0)

**问：** 如何让 Claude Code (Anthropic 原生客户端) 零感知接入 Blend？

**答：** 实施 "协议劫持与意图解构" 策略：

1. **"木马"伪装 (The Trojan Masking)：**
   - **决策：** `/v1/models` 接口必须硬编码返回 `claude-3-5-sonnet-20241022` 等 Anthropic 标准 ID。
   - **原因：** Claude Code 内部存在针对特定模型版本号的功能分支检查（如 Tools/Caching 支持）。若返回 `blend` 等自定义 ID，客户端会回退至功能缺失的"文本模式"。

2. **协议桥接与事件转换 (Protocol Bridging)：**
   - **决策：** 建立 SSE 转换层。将 Blend 内部的 OpenAI-style 异步 Chunk 实时翻译为 Anthropic 的 `message_start`, `content_block_delta`, `message_stop` 事件序列。
   - **优化：** 修正了 Chunk 结构的顶层映射，支持 `tool_use` 协议的透传，确保 Blend 后端能完美驱动 Claude Code 的本地文件操作与命令执行。

3. **任务感知型 Tier 映射 (Task-Aware Mapping)：**
   - **决策：** 将客户端请求的模型名称视为"意图元数据"而非物理指令。
     - 请求 `sonnet` -> 触发 **自动赛跑 (Auto-Race)** 模式（默认最优性价比）。
     - 请求 `opus` -> 强力引导至 **L5 深度质检 + 强制反馈循环**（追求极致质量）。
     - 请求 `haiku` -> 开启 **极速压榨模式**（关闭部分 L5 检查以换取秒回体验）。

**原因：** Blend 的定位是"智能协议转换器"。通过在接口层伪装成 Anthropic，在执行层解构为 5 层自愈流水线，实现了**"原生客户端的体验 + Blend 的成本控制"**。这证明了 Blend 架构对商用复杂工程工具（如 Claude Code）的深度支撑能力。
