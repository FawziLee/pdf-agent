# PDF Agent

上传 PDF → 自动理解文档内容 → 用自然语言提问。

通用 PDF 智能问答助手：先解析、摘要并向量化入库，再基于原文检索回答。
---

## 先记住这几件事（最重要）

| 优先级 | 要点 |
|--------|------|
| 1 | **必须先配置自己的密钥**：复制 `.env.example` → `.env`，填入 `QWEN_API_KEY`、`PADDLE_TOKEN`。没有 key 无法 OCR / 摘要 / 问答。 |
| 2 | **首次上传会比较慢**：要跑 OCR + 多章摘要 + Embedding，属于正常现象。 |

---

## 30 秒上手（推荐路径）

```bash
# 1. 进入项目
cd pdf-agent

# 2. 配置密钥（必做）
cp .env.example .env
# 编辑 .env，填入你自己的 QWEN_API_KEY、PADDLE_TOKEN

# 3. 启动
docker compose up -d --build

# 4. 打开界面
# Gradio → http://localhost:7860
# API 文档 → http://localhost:5557/docs
```

启动后在 Gradio 里：**上传 PDF → 等待入库完成 → 提问**。

---

## 它能做什么

1. **上传 PDF** — OCR 识别版面与文字（扫描件 / 图片型 PDF 也能用）
2. **生成摘要** — 章节摘要 + 全文总摘要，写入 SQLite
3. **向量入库** — 语义检索（dense）+ 关键词检索（BM25，中文 jieba）
4. **流式问答** — 先返回检索块（页码、坐标），再流式生成回答
5. **Gradio 界面** — 浏览器里完成上传、检索可视化、对话

---

## 整体流程（通俗版）

```text
你上传一份 PDF
        │
        ▼
   OCR 读出文字和版面结构
        │
        ├─► 大模型写章节摘要 / 全文摘要  → 存进 SQLite
        │
        └─► 切成文本块 + 向量化          → 存进 Milvus
        │
        ▼
你提问：「第三章的主要结论是什么？」
        │
        ▼
   向量检索 + BM25 找出相关段落
        │
        ▼
   大模型只根据这些原文回答（可引用页码）
```

核心原则：**先「读懂并记住」文档，再「带着原文」回答问题。**

---

## 密钥配置（必读）

| 变量 | 是否必填 | 用途 |
|------|----------|------|
| `QWEN_API_KEY` | **必填** | 通义千问：摘要、问答、Embedding |
| `PADDLE_TOKEN` | **必填** | PaddleOCR：PDF 版面解析 |
| `LLM_MODEL_NAME` | 可选 | 默认 `qwen-plus` |
| `EMBEDDING_MODEL_NAME` | 可选 | 默认 `qwen3.7-text-embedding` |

```bash
cp .env.example .env
# 再编辑 .env
```
---

## 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| **Gradio 界面** | http://localhost:7860 | 日常使用入口 |
| API 文档 | http://localhost:5557/docs | 接口调试 |
| 健康检查 | http://localhost:5557/health | 服务是否正常 |

---

## Docker 日常命令

```bash
docker compose ps               # 看是否在跑
docker compose logs -f          # 看日志（排错首选）
docker compose logs -f api      # 只看后端
docker compose down             # 停止
docker compose up -d --build    # 改代码后重新构建并启动
```

### 数据存在哪里（重要）

```text
./data/
├── pdf_agent.db      # 文档元数据、摘要
├── milvus.db/        # 向量与 BM25 索引
└── upload/           # 上传的 PDF 副本

./result/             # OCR 等调试结果
```

- Compose 已挂载 `./data`、`./result` 到容器
- **删容器 ≠ 丢数据**；删 `./data` 才会清空

---

## 本地开发启动（不用 Docker）

需要：Python ≥ 3.12、[uv](https://github.com/astral-sh/uv)

```bash
cd pdf-agent
uv sync

# 终端 1：API
uv run python main.py          # http://0.0.0.0:5557

# 终端 2：Gradio
uv run python ui/gradio_app.py # http://0.0.0.0:7860
```

本地同样需要配置好 `QWEN_API_KEY`、`PADDLE_TOKEN`（`.env` 或 shell 环境变量均可）。

---

## Gradio 使用步骤

1. 打开 http://localhost:7860  
2. 左侧上传 PDF → 点「上传并入库」（**首次较慢，请等完成**）  
3. 确认文档列表里的 `document_id`，需要时可填入「限定文档」  
4. 右侧输入问题并发送  
5. 下方 Tab 查看检索依据：
   - **向量检索 dense**
   - **BM25 检索**
   - **合并结果**（真正交给大模型的上下文）

检索块会展示：**完整文本、页码、坐标 bbox、分数、来源**。

---

## 主要 API

完整文档：http://localhost:5557/docs

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/documents/upload` | **上传并入库（问答前先做这一步）** |
| `GET` | `/api/v1/documents/list` | 文档列表 |
| `GET` | `/api/v1/documents/{document_id}` | 文档详情 |
| `DELETE` | `/api/v1/documents/{document_id}` | 删除文档 |
| `POST` | `/api/v1/agents/chat` | **流式问答（SSE）** |
| `GET` | `/health` | 健康检查 |

### 上传示例

```bash
curl -X POST "http://localhost:5557/api/v1/documents/upload?tenant_id=demo-tenant&user_id=demo-user" \
  -F "file=@/path/to/your.pdf"
```

### 流式问答示例

```bash
curl -N -X POST "http://localhost:5557/api/v1/agents/chat?tenant_id=demo-tenant" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "question": "这份文档的主要内容是什么？",
    "document_id": "你的document_id",
    "top_k": 5,
    "use_bm25": true
  }'
```

SSE 事件顺序：`contexts`（检索）→ `delta`（回答增量）→ `done`；出错时为 `error`。

---

## 技术栈

| 模块 | 技术 |
|------|------|
| API | FastAPI + Uvicorn |
| 界面 | Gradio |
| OCR | PaddleOCR（云端 API） |
| 大模型 / Embedding | 通义千问（DashScope 兼容接口） |
| 元数据 | SQLite |
| 向量库 | Milvus Lite（本地文件，含 BM25） |
| 依赖管理 | uv |

---

## 项目结构

```text
pdf-agent/
├── app/                  # 后端核心
│   ├── api/v1/           # HTTP 接口
│   ├── agent.py          # PdfAgent：摘要 / 入库 / 检索 / 问答
│   ├── core/             # OCR、Embedding、LLM、向量库
│   ├── db/               # SQLite、Milvus
│   ├── models/           # 数据表
│   └── service/          # 上传与业务逻辑
├── ui/gradio_app.py      # Gradio 前端
├── scripts/              # 调试脚本
├── data/                 # 运行时数据（重要，勿随意删除）
├── result/               # OCR / 调试输出
├── main.py               # FastAPI 入口
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

---

本项目为演示用途。请遵守你所使用的 OCR / LLM / Embedding 服务商的使用条款与配额限制。
