# Blend

极致成本效率商用 API — 质量 > Claude Sonnet 4.6，成本优化 (v2.3.0)

## 特性

- **4 层自愈架构**: L1 (草稿+评分) → L2 (策略) → L3 (执行) → L5 (纠错+终审)
- **Draft-Refine 模式**: 极致压榨 Minimax (几乎免费) 生成初版草稿，精英模型仅负责核心精修。
- **不可摧毁的心脏**: 
  - **持久化记忆**: 重启后依然避开故障 Provider。
  - **赛跑式 Fallback**: 并行探测多模型，3秒切换，用户无感知。
  - **自愈纠错**: 质检失败自动触发反馈重写循环。
- **统筹计费优化**: 自动平衡"按量" (Baosi) 与"按次" (Lemon) 计费，永远选择 ROI 最高路径。
- **OpenAI & Anthropic 兼容**: 完美对接 Claude Code 和各种主流 AI 客户端。

## 安装

```bash
# 克隆项目并安装
cd blend
pip install -e .

# 复制环境变量模板并填入 Key
cp .env.example .env
```

## 配置

```bash
# API Keys
MINIMAX_API_KEY=...
BAOSI_API_KEY=...
LEMON_API_KEY=...
```

## 使用

### 启动 API 服务
```bash
python3 -m uvicorn blend.api:app --host 0.0.0.0 --port 8000
```

### 检查系统健康
```bash
curl http://localhost:8000/health
```

## 架构

```
用户请求
    ↓
L1: 复杂度评分 + 意图检测
    ↓
L2: 策略生成 (仅 HIGH 复杂度)
    ↓
L3: Recipe 执行 (DRAFT → REFINE → VERIFY)
    ↓
L5: 质量验证 + 自愈纠错
    ↓
优雅交付
```

## Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f blend

# 停止服务
docker-compose down
```

## 许可证

MIT
