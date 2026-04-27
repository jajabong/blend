# Blend 项目审计 Prompt

## 角色
你是一个专业的代码审计员，同时具备 TDD 开发能力和自动化测试验证能力。你将全面审计 blend 项目的代码质量、安全性、性能和架构设计。

## 目标
对 `/Users/dongshenglu/blend` 项目进行**全自动**审计，包括：
1. 代码分析（静态检查 + 动态分析）
2. TDD 验证（识别缺失测试 → 编写测试 → 验证通过）
3. 自动化测试执行（单元测试 + 集成测试 + 覆盖率分析）

## 执行流程

### Phase 1: 环境检查
```bash
cd /Users/dongshenglu/blend
source .venv/bin/activate
python --version
ruff check . --output-format=concise
mypy . --ignore-missing-imports
```

### Phase 2: 代码结构审计
1. 扫描所有 Python 文件
2. 检查：
   - [ ] 类型注解完整性
   - [ ] 错误处理覆盖
   - [ ] 硬编码凭证检查
   - [ ] 依赖项安全性
   - [ ] 代码复杂度（圈复杂度）

### Phase 3: TDD 验证
对每个核心模块执行 TDD 循环：
1. 识别缺失测试的函数
2. 编写失败测试（RED）
3. 实现功能（GREEN）
4. 重构优化（IMPROVE）
5. 验证覆盖率 ≥ 80%

### Phase 4: 自动化测试执行
```bash
# 单元测试
pytest tests/ -v --cov=blend --cov-report=term-missing --cov-fail-under=80

# 集成测试
pytest tests/test_integration.py -v

# Smoke 测试
blend status
```

### Phase 5: 安全扫描
```bash
# 凭证检查
grep -r "sk-\|api_key\|password\|secret" --include="*.py" blend/ | grep -v ".venv"

# 依赖安全
pip audit || safety check
```

### Phase 6: 性能分析
- 检查 L1-L5 各层执行时间
- 验证 Token 消耗合理性
- 检查预算追踪准确性

## 输出格式

### 审计报告结构
```markdown
# Blend 项目审计报告

## 1. 环境状态
- Python 版本
- 依赖完整性
- Lint/Type 检查结果

## 2. 代码质量评分
| 模块 | 覆盖率 | 类型检查 | 复杂度 |
|------|--------|----------|--------|
| ... | ... | ... | ... |

## 3. 安全问题
- [ ] 问题1
- [ ] 问题2

## 4. TDD 验证
- 缺失测试数
- 新增测试数
- 覆盖率变化

## 5. 测试结果
- 单元测试通过率
- 集成测试通过率
- Smoke 测试结果

## 6. 建议
- 优先级 P0
- 优先级 P1
- 优先级 P2

## 7. 执行摘要
- 总耗时
- 发现问题数
- 修复问题数
- 当前覆盖率
```

## 执行命令

在 opencode 中执行以下命令启动审计：

```
请对 /Users/dongshenglu/blend 项目执行全自动审计：
1. 先运行 ruff check . 和 mypy . 检查代码质量
2. 运行 pytest tests/ -v --cov=blend --cov-report=term-missing 验证测试覆盖率
3. 检查安全漏洞：grep -r "sk-\|api_key" blend/ | grep -v test
4. 输出完整的审计报告，包括代码质量评分、安全问题、TDD 验证结果
5. 对发现的问题自动修复，并验证修复效果

开始执行！
```

## 关键检查点

### 必须通过的检查
- [ ] ruff check . → 无 ERROR
- [ ] mypy . → 无 ERROR
- [ ] pytest tests/ → 全绿
- [ ] 覆盖率 ≥ 80%
- [ ] 无硬编码凭证
- [ ] blend status → 全绿

### 自动修复的问题类型
1. 类型注解缺失 → 自动添加
2. 错误处理缺失 → 添加 try/except
3. 测试覆盖率不足 → 补充测试
4. 命名不规范 → 自动重命名

## 约束条件
- 只修改测试文件，不修改核心业务逻辑（除非发现严重 bug）
- 修复前先确认，修复后验证
- 保持代码风格一致性
- 所有修改通过 ruff format 自动格式化

## 成功标准
1. ruff + mypy + pytest 全绿
2. 覆盖率 ≥ 85%
3. 无安全漏洞
4. 审计报告完整

---
Generated for opencode with Sisyphus orchestration
