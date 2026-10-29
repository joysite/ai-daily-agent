# AI Daily Agent（本地模型版）

一个可独立运行的每日 AI 热点新闻 Agent，采用本地 Ollama 模型完成新闻筛选和摘要，通过 Chroma 进行语义去重，使用 SQLite 持久化新闻与运行记录，最终生成 Markdown 格式的每日 AI 新闻简报。

处理流程：

`RSS 采集 → 正文提取 → 本地 LLM 筛选 → 新闻摘要 → Chroma 语义去重 → SQLite 持久化 → Markdown 日报`

## 功能与边界

- **新闻采集：** 通过 RSS 获取新闻，尽可能使用 Trafilatura 提取正文。新闻源网站需要联网。
- **本地推理：** 新闻筛选、摘要和 Embedding 均通过本机 Ollama 完成，不调用云端 LLM API。
- **数据持久化：** SQLite 保存新闻与每次运行记录，相同 URL 重复运行时更新已有记录，避免无限新增。
- **语义去重：** 使用 Chroma 持久化语义向量，默认检索最近 7 天的新闻。
- **异常处理：** 某个 RSS 源失败时记录日志并继续处理其他来源。
- **Embedding 降级：** Embedding 服务失败时默认采用 fail-open 策略，保留新闻，避免整个日报流程因向量服务异常而中断。
- **自动化测试：** 支持针对新闻筛选、数据库和主流程集成逻辑运行测试。
- **可观测性：** 记录运行状态、新闻筛选结果及异常信息，便于定位模型输出和处理流程的问题。

## 1. 环境准备（Ubuntu）

建议使用 Python 3.10 或更高版本，并安装 Ollama。

检查环境：

```bash
python3 --version
ollama --version
```

下载所需模型：

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama list
```

其中：

- `qwen2.5-coder:7b`：本地新闻分析与文本生成。
- `nomic-embed-text`：新闻语义向量生成。

## 2. 安装

解压项目并进入目录：

```bash
unzip ai_daily_agent_complete.zip
cd ai_daily_agent_complete
```

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

根据项目实际配置检查 `settings.yaml` 和 `.env`。

确认 Ollama 服务可用：

```bash
curl http://localhost:11434/api/tags
```

如果 Ollama 服务没有启动，可以在终端执行：

```bash
ollama serve
```

如果服务已由系统或其他进程启动，则无需重复启动。

## 3. 运行

在项目根目录执行：

```bash
PYTHONPATH=. python -m src.main
```

运行结束后，主要产物包括：

| 路径 | 用途 |
|---|---|
| `output/YYYY-MM-DD.md` | 每日日报 |
| `data/news.db` | SQLite 新闻与运行记录 |
| `data/chroma/` | Chroma 持久化向量数据库 |
| `logs/ai_daily_agent.log` | 程序运行日志 |

具体路径以项目当前配置为准。

## 4. 新闻筛选与模型输出约束

新闻筛选规则集中在 `prompts/filter_news.txt`，筛选逻辑位于 `src/llm_processor.py`，Ollama 调用与 JSON Schema 约束位于 `src/ollama_client.py`。

### 4.1 筛选原则

新闻筛选应综合考虑新闻的核心事件、实际进展、信息完整度和参考价值，而不是只看新闻所属类别。

- **模型：** 关注新模型发布、重要升级和有依据的能力改进。
- **论文：** 关注研究问题、方法创新、技术针对性、潜在应用和已有证据。不能仅因缺少准确率、数据集或性能提升数值就直接判定为低价值。
- **产品：** 关注 AI 工具的新功能、实际应用价值及成本或开发能力变化，不能因为没有基础模型训练或学术论文就降低评分。
- **融资：** 关注融资金额、资金用途及其对研发或商业化的意义。
- **行业：** 关注重要政策、监管变化、公共服务、数据保护、人工审核、责任机制、基础设施和产业合作等。不能仅因新闻属于政策而不是模型发布、产品发布或融资就直接过滤。

### 4.2 评分与最低收录分数

当前配置中的最低收录分数为：

```yaml
filter:
  min_score: 6
```

评分的基本含义：

| 分数 | 判断标准 |
|---|---|
| 1–3 | AI 无关、教程、广告、重复旧闻或空泛且缺乏实质信息 |
| 4–5 | AI 相关，但新进展不明确、信息增量较小或参考价值有限 |
| 6–7 | 有可识别的新进展，对开发者、研究者或行业有参考价值 |
| 8–9 | 重要模型发布、值得关注的研究成果、具有实际价值的产品更新、重大融资或重要行业政策 |
| 10 | 具有重大影响，且有充分信息和证据支持 |

评分应结合新闻实际内容判断。缺少某项量化指标，不等于新闻没有价值；同样，也不能仅凭“重大突破”等宣传性措辞给出高分。

### 4.3 单条筛选与结构化输出

当前筛选流程采用 `batch_size = 1`，即逐条处理新闻，以降低多条新闻同时输入模型时出现事实、实体或理由混用的风险。

筛选结果通过 JSON Schema 约束结构，主要字段包括：

- `index`：对应输入新闻的索引。
- `keep`：是否保留新闻。
- `score`：1–10 分的评分。
- `category`：新闻分类。
- `reason`：筛选理由。

Python 端会检查必要字段、字段类型、评分范围和分类是否合法。对于不符合要求的结果，程序不会直接将其作为已审核通过的新闻放行。

**注意：** JSON Schema 只能约束输出结构，不能保证理由与原文完全一致，也不能保证分类和评分在语义上正确。因此，结构化输出校验不等于事实核验。

## 5. 查看数据库

查看新闻总量、非重复新闻数量及重复新闻数量：

```bash
sqlite3 data/news.db \
'SELECT COUNT(*) AS total,
        SUM(is_duplicate=0) AS unique_count,
        SUM(is_duplicate=1) AS duplicate_count
 FROM news;'
```

查看最近 5 次运行记录：

```bash
sqlite3 data/news.db \
'SELECT run_id,status,collected_count,processed_count,
        unique_count,duplicate_count,report_path
 FROM runs
 ORDER BY run_id DESC
 LIMIT 5;'
```

若系统未安装 `sqlite3` CLI，可以使用 Python 查询；程序本身不依赖 sqlite3 命令行工具。

## 6. RSS 源与时间窗口

默认采集最近 24 小时的新闻，具体时间窗口由 `settings.yaml` 配置。

RSS 源地址可能发生变化，也可能短时不可用。程序会记录来源采集成功或失败的日志，并在单个来源失败时继续处理其他来源。

可以在 `settings.yaml` 的 `collector.feeds` 中添加、修改或关闭新闻源。

## 7. 语义去重阈值调优

`cosine_distance_threshold` 用于控制语义去重判定。对于常见的余弦距离定义，阈值越大，越容易将两条新闻判定为相似内容。

默认值 `0.18` 是初始参数，并非适用于所有新闻源的通用最优值。

- 重复新闻漏检明显：可以逐步尝试 `0.22`、`0.25`。
- 不同事件被误合并：可以尝试 `0.12`、`0.15`。

建议先人工检查 30–50 条新闻的去重结果，再根据误判情况调整。每次调整只改变一个参数，并保留调整前后的样本结果，避免单凭日报条数判断去重质量。

## 8. 测试

### 8.1 运行自动化测试

运行新闻筛选、数据库及主流程集成测试：

```bash
python -m pytest -v \
tests/test_llm_filter.py \
tests/test_database.py \
tests/test_main_integration.py
```

也可以运行项目原有的 unittest 测试：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

两种命令的测试范围可能不同，应以实际发现并执行的测试用例为准。

### 8.2 新闻筛选离线测试

当前离线筛选测试覆盖结构化结果处理、异常或无效字段处理，以及最低分数边界等情况。

已确认的测试结果：

- `tests/test_llm_filter.py`：12 项通过。
- 此前合并运行 `tests/test_llm_filter.py`、`tests/test_database.py` 和 `tests/test_main_integration.py`：16 项通过。
- 新增的最低分数边界测试确认：当新闻评分恰好为 6，且模型将其标记为保留时，新闻应通过最低分数门槛。

离线测试使用模拟模型输出，主要验证 Python 端的筛选逻辑，不能证明真实 Ollama 模型一定会遵守提示词。

### 8.3 真实本地模型测试

使用项目中的真实模型筛选测试脚本：

```bash
python -m tests.test_llm_filter_live
```

该脚本使用 5 条人工合成新闻测试本地模型，包括论文、模型发布、AI 产品、融资和公共政策，不需要运行完整新闻采集流程。

**最近一次真实模型测试的结果：**

| 新闻类型 | 模型评分 | 结果 | 发现的问题 |
|---|---:|---|---|
| 小目标检测论文 | 4 | 过滤 | 仍过度依赖量化实验指标 |
| 大语言模型发布 | 8 | 保留 | 理由声称存在明确能力改进，但测试新闻没有提供相关证据 |
| AI 文档工具 | 6 | 保留 | 本次判断与新闻提供的功能信息基本一致 |
| AI 初创公司融资 | 8 | 保留 | 理由与融资金额、用途信息基本一致 |
| 公共服务 AI 政策 | 4 | 过滤 | 仍存在将公共政策错误视为低价值新闻的倾向 |

以上是特定测试样本的结果，不代表模型在所有新闻上的整体准确率。

## 9. MCP 服务：自然语言检索历史 AI 新闻

项目提供基于 Model Context Protocol（MCP）的本地服务，可以将已有新闻数据库和 Chroma 语义检索能力暴露给支持 MCP 的客户端，例如 Claude Desktop 和 Cursor。

MCP 服务是现有新闻 Agent 的扩展，不会替代原有新闻采集、筛选、去重和日报生成流程。

### 9.1 提供的工具

| 工具 | 用途 | 参数 |
|---|---|---|
| `query_news_stats` | 查询新闻总量、重复新闻数量、平均评分、分类统计和最近任务运行记录 | 无 |
| `search_news_semantic` | 根据自然语言查询语义检索历史新闻 | `query`：查询文本；`days`：检索时间窗口，默认 7 天，范围 1–365 天 |

语义检索最多返回 10 条结果。返回字段包括新闻标题、链接、来源、入库时间、文本内容和向量距离。

**时间字段说明：** 当前语义检索的 `days` 参数根据 Chroma 中记录的 `created_epoch`（入库时间）筛选，不代表新闻的实际发布时间。返回的 `distance` 是向量距离，不是相似度百分比；距离越小通常表示向量越接近。

### 9.2 环境要求

启动前请确认：

- 已按照前文安装项目依赖。
- 已安装并启动 Ollama。
- 已下载 `qwen2.5-coder:7b` 和 `nomic-embed-text` 模型。
- 项目配置文件 `settings.yaml` 中的 Ollama 地址、SQLite 路径和 Chroma 路径正确。
- 至少运行过一次新闻采集流程，且已有可供查询的数据。

MCP 服务使用现有 SQLite 数据库和 Chroma 向量库，不需要另外创建一套新闻数据。

### 9.3 启动 MCP 服务

进入项目根目录，激活已安装项目依赖的 Python 环境，然后执行：

```bash
python -m mcp_server.server
```

### 9.4 Claude Desktop 配置示例

在 Claude Desktop 的 MCP 配置文件中，将以下配置合并到已有的 `mcpServers` 对象中。请勿覆盖其他 MCP 服务配置。

```json
{
  "mcpServers": {
    "ai-daily-agent": {
      "command": "/绝对路径/ai-daily-agent/bin/python",
      "args": [
        "-m",
        "mcp_server.server"
      ]
    }
  }
}
```

### 9.5 Cursor 配置说明

在 Cursor 的 MCP 设置中添加本地 MCP 服务。具体设置入口可能随 Cursor 版本变化，请以当前版本的设置界面为准。

如果使用 JSON 配置，可参考以下结构，并将 `command` 替换为当前 Python 环境的实际绝对路径：

```json
{
  "mcpServers": {
    "ai-daily-agent": {
      "command": "/绝对路径/ai-daily-agent/bin/python",
      "args": [
        "-m",
        "mcp_server.server"
      ]
    }
  }
}
```



### 9.6 自然语言调用示例

完成 MCP 客户端配置并确认服务连接成功后，可以直接使用自然语言调用工具。

**查询新闻数据库统计：**

- 查询当前新闻总量、重复新闻数量和平均评分。
- 统计不同新闻类别的数量。
- 查看最近几次新闻采集任务的运行状态。

**语义检索历史新闻：**

- 搜索最近 7 天与大语言模型和多智能体系统相关的新闻。
- 查找最近 30 天的计算机视觉研究进展。
- 检索与 AI Agent 工具调用、长期记忆相关的历史报道。

检索结果包含新闻标题、链接、来源和相关文本，可用于进一步阅读和整理。

注意：自然语言示例是面向 MCP 客户端的使用方式，并不意味着服务会自动执行新闻采集或修改数据库。语义检索的时间窗口依据当前实现中的入库时间字段筛选。



### 9.7 数据访问与安全边界

- **SQLite 只读访问：** MCP 统计工具通过只读方式访问现有 SQLite 数据库，不负责创建、修改或删除新闻记录。
- **Chroma 语义检索：** 语义检索读取现有 `ai_news` 集合，并调用本地 Ollama 生成查询向量，不负责新增或删除向量记录。
- **本地运行：** 当前服务使用 MCP 的 `stdio` 传输方式，由 MCP 客户端启动并通信，不是面向公网的 HTTP API。
- **本地权限：** `stdio` 不等于身份认证。能够启动该服务的客户端进程通常也能以当前操作系统用户的权限访问相关文件，请妥善管理本机账户和文件权限。
- **避免公开敏感数据：** 不要将 `.env`、访问令牌、个人凭据、SQLite 数据库、Chroma 数据目录及包含敏感信息的日志提交到公开仓库。
- **配置检查：** 开源前请检查 `.gitignore`，确认本地数据、密钥、运行日志和生成的日报不会被意外提交。

以上是当前 MCP 服务的设计边界。若后续增加写入工具、远程传输或多用户访问，需要重新评估权限控制、输入校验和身份认证。


## 10. 每天自动运行（可选）

先确认手动运行成功，再编辑定时任务：

```bash
crontab -e
```

添加类似配置，按实际项目目录修改：

```cron
15 8 * * * cd /绝对路径/ai_daily_agent_complete && /绝对路径/ai_daily_agent_complete/.venv/bin/python -m src.main >> logs/cron.log 2>&1
```

该任务按运行机器的系统时区每天 08:15 执行。日报日期使用哪个时区，则以 `settings.yaml` 的配置为准。

## 11. 使用建议

- 首次运行时，先检查 RSS 采集、Ollama 推理、Embedding、数据库写入和日报生成是否正常。
- 优先使用小规模测试验证提示词变更，不要在未确认筛选结果之前直接执行完整采集流程。
- 调整筛选规则时，保留固定测试样本，比较评分、分类、理由和最终保留结果。
- 检查日志时，重点关注来源失败、模型输出异常、新闻过滤原因和去重异常。
- 在确认本地模型筛选质量之前，建议人工抽查日报，尤其是论文、政策和新模型发布类新闻。
