# Blend

极致成本效率商用 API — 质量 > Claude Opus 4.7，成本 < 808 CNY/月（v1.3.2）

## 特性

- **5 层架构**: L1 压缩 → L2 策略 → L3 执行 → L4 压缩 → L5 终审
- **动态模型选择**: 复杂度 1-2 → Minimax, 3-5 → Haiku/Sonnet, 6-10 → Sonnet (+ L2 Opus 策略)
- **Opus 策略生成**: HIGH 复杂度任务自动调用 Opus 生成执行计划
- **OpenAI 兼容**: 支持 `/v1/chat/completions` 等标准端点
- **Token 优化**: 自动压缩提示词和输出，节省成本
- **预算追踪**: 实时 API 预算状态监控

## 安装

```bash
# 克隆项目
cd blend

# 安装依赖
pip install -e .

# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
```

## 配置

编辑 `.env` 文件：

```bash
# API Keys (必需)
MINIMAX_API_KEY=your_minimax_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
LEMON_API_KEY=your_lemonapi_key_here

# 可选配置
PORT=8000
LOG_LEVEL=INFO
```

### API Key 获取

| 服务 | 获取地址 |
|------|---------|
| Minimax | https://platform.minimax.chat/ |
| Anthropic | https://console.anthropic.com/ |
| LemonAPI | https://lemonapi.site/ (第三方 Claude 代理) |

## 使用

### 检查状态

```bash
blend status
```

### 启动服务

```bash
blend serve
```

服务将在 http://localhost:8000 启动。

### API 调用示例

#### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "haiku",
    "messages": [{"role": "user", "content": "Hello, explain distributed systems in 3 sentences"}]
  }'
```

#### Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "haiku",
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

#### 查看可用模型

```bash
curl http://localhost:8000/v1/models
```

#### 查看预算状态

```bash
curl http://localhost:8000/v1/budget
```

#### 查看系统信息

```bash
curl http://localhost:8000/v1/info
```

## 响应格式

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "haiku",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "响应内容..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "_blend_metadata": {
    "complexity": 3,
    "layer_path": "L1>L3>L5",
    "model_used": "minimax",
    "tokens_used": 120,
    "quality_gate_passed": true,
    "l1_compressed": true,
    "l4_applied": false
  }
}
```

### Blend 元数据说明

| 字段 | 说明 |
|------|------|
| `complexity` | 复杂度评分 (1-10) |
| `layer_path` | 经过的层级路径 |
| `model_used` | 实际使用的模型 |
| `tokens_used` | LLM 消耗的 token 数 |
| `quality_gate_passed` | L5 质量门是否通过 |
| `l1_compressed` | 是否经过 L1 压缩 |
| `l4_applied` | 是否经过 L4 压缩 |

## 错误处理

### 401 Unauthorized

API Key 未设置或无效：

```json
{"detail": "MINIMAX_API_KEY is required"}
```

### 400 Bad Request

请求格式错误：

```json
{"detail": "Messages cannot be empty"}
```

### 500 Internal Server Error

处理出错：

```json
{"detail": "Processing error: ..."}
```

## 架构

```
用户请求
    ↓
L1: 压缩 + 复杂度评分 (Minimax)
    ↓
L2: 策略生成 (Opus, 仅 HIGH 复杂度)
    ↓
L3: 执行 (动态模型选择)
    ↓
L4: 输出压缩 (Minimax, >500T 时触发)
    ↓
L5: 质量终审
    ↓
返回响应
```

## 开发

```bash
# 运行测试
pytest tests/ -v --cov=blend

# 运行快速测试（跳过真实 API 调用）
pytest tests/ -v -m "not real_api"

# Lint 检查
ruff check .

# 类型检查
mypy .
```

## 测试

测试覆盖：93%（209 passed, 1 skipped）

- **Mocked tests**: 快速确定性测试，provider.chat() 被 mock
- **Real API tests**: 需标记 `-m real_api` 显式启用（慢，需网络）
- **Integration tests**: `test_real_integration.py` 包含完整管道测试

## 许可证

MIT
