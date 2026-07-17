"""Gradio 前端：对接 FastAPI 上传 / 列表 / 流式问答接口。

先启动后端：
  uv run python main.py

再启动本界面：
  uv run python ui/gradio_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import httpx

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5557")
DEFAULT_GRADIO_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
UPLOAD_TIMEOUT = 600.0
CHAT_TIMEOUT = 300.0


def _api_base(base_url: str) -> str:
    return (base_url or DEFAULT_BASE_URL).rstrip("/")


def _doc_upload_url(base_url: str) -> str:
    return f"{_api_base(base_url)}/api/v1/documents/upload"


def _doc_list_url(base_url: str) -> str:
    return f"{_api_base(base_url)}/api/v1/documents/list"


def _chat_url(base_url: str) -> str:
    return f"{_api_base(base_url)}/api/v1/agents/chat"


def _format_bbox(bbox: Any) -> str:
    if bbox is None or bbox == "":
        return "无"
    if isinstance(bbox, (list, tuple)):
        return json.dumps(list(bbox), ensure_ascii=False)
    if isinstance(bbox, str):
        try:
            parsed = json.loads(bbox)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            return bbox
    return str(bbox)


def _format_page(page_idx: Any) -> str:
    if page_idx is None:
        return "未知"
    try:
        # OCR 页码从 0 开始，界面展示为第 N 页
        return f"第 {int(page_idx) + 1} 页"
    except (TypeError, ValueError):
        return str(page_idx)


def _format_hit(hit: dict[str, Any], index: int, tag: str) -> str:
    text = (hit.get("text") or "").strip()
    return (
        f"[{tag}] #{index}\n"
        f"分数: {hit.get('score')}\n"
        f"页码: {_format_page(hit.get('page_idx'))} (page_idx={hit.get('page_idx')})\n"
        f"坐标 bbox: {_format_bbox(hit.get('bbox'))}\n"
        f"document_id: {hit.get('document_id')}\n"
        f"tenant_id: {hit.get('tenant_id')}\n"
        f"-----\n"
        f"{text}"
    )


def _format_hit_list(hits: list[dict[str, Any]], tag: str, empty_msg: str) -> str:
    if not hits:
        return empty_msg
    return "\n\n====================\n\n".join(
        _format_hit(hit, i, tag) for i, hit in enumerate(hits, 1)
    )


def _split_retrieval_views(data: dict[str, Any] | list[dict[str, Any]]) -> tuple[str, str, str]:
    if isinstance(data, list):
        dense, bm25, merged = data, [], data
        bm25_error = None
        bm25_enabled = True
    else:
        dense = data.get("dense") or []
        bm25 = data.get("bm25") or []
        merged = data.get("contexts") or []
        bm25_error = data.get("bm25_error")
        bm25_enabled = data.get("bm25_enabled", True)

    dense_text = _format_hit_list(
        dense,
        tag="向量检索 dense",
        empty_msg="[向量检索 dense] 无结果",
    )

    if not bm25_enabled:
        bm25_text = "[BM25] 未开启（请勾选「混合 BM25」）"
    elif bm25_error:
        bm25_text = f"[BM25] 检索失败\n{bm25_error}"
    else:
        bm25_text = _format_hit_list(
            bm25,
            tag="BM25",
            empty_msg="[BM25] 无结果。中文查询需使用 chinese/jieba 分词；若刚升级 schema，请重新上传 PDF 后再试。",
        )

    merged_text = _format_hit_list(
        merged,
        tag="合并结果",
        empty_msg="[合并结果] 无结果",
    )
    return dense_text, bm25_text, merged_text


def _format_contexts(data: dict[str, Any] | list[dict[str, Any]]) -> str:
    dense_text, bm25_text, merged_text = _split_retrieval_views(data)
    return (
        "======== 向量检索 (dense) ========\n"
        f"{dense_text}\n\n"
        "======== BM25 检索 ========\n"
        f"{bm25_text}\n\n"
        "======== 合并去重（用于回答） ========\n"
        f"{merged_text}"
    )


def _format_document_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return "暂无文档，请先上传 PDF"

    lines = [
        "| 文件名 | document_id | 状态 | 向量块数 |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(
            f"| {item.get('file_name', '')} "
            f"| `{item.get('document_id', '')}` "
            f"| {item.get('status', '')} "
            f"| {item.get('chunk_count', '')} |"
        )
    return "\n".join(lines)


def _fetch_documents(base_url: str, tenant_id: str) -> tuple[list[dict[str, Any]], str]:
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                _doc_list_url(base_url),
                params={"tenant_id": tenant_id or "demo-tenant", "page": 1, "page_size": 50},
            )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        return [], f"刷新失败：{exc}"

    if payload.get("code") != 200:
        return [], f"刷新失败：{payload.get('message')}"

    items = (payload.get("data") or {}).get("list") or []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            item = {
                "document_id": getattr(item, "document_id", None),
                "file_name": getattr(item, "file_name", None),
                "status": getattr(item, "status", None),
                "chunk_count": getattr(item, "chunk_count", None),
            }
        doc_id = item.get("document_id") or ""
        if doc_id:
            normalized.append(item)
    msg = f"共 {len(normalized)} 份文档" if normalized else "暂无文档，请先上传 PDF"
    return normalized, msg


def refresh_documents(
    base_url: str,
    tenant_id: str,
) -> tuple[str, str, str]:
    items, msg = _fetch_documents(base_url, tenant_id)
    table_md = _format_document_table(items)
    first_id = items[0].get("document_id", "") if items else ""
    return table_md, msg, first_id


def upload_document(
    file_path: str | None,
    base_url: str,
    tenant_id: str,
    user_id: str,
) -> tuple[str, str, str, str]:
    if not file_path:
        table_md, msg, doc_id = refresh_documents(base_url, tenant_id)
        return table_md, msg, doc_id, "请先选择 PDF 文件"

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        table_md, msg, doc_id = refresh_documents(base_url, tenant_id)
        return table_md, msg, doc_id, "仅支持 PDF 文件"

    try:
        with httpx.Client(timeout=UPLOAD_TIMEOUT) as client:
            with path.open("rb") as f:
                resp = client.post(
                    _doc_upload_url(base_url),
                    params={
                        "tenant_id": tenant_id or "demo-tenant",
                        "user_id": user_id or "demo-user",
                    },
                    files={"file": (path.name, f, "application/pdf")},
                )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        table_md, msg, doc_id = refresh_documents(base_url, tenant_id)
        return table_md, msg, doc_id, f"上传失败：{exc}"

    if payload.get("code") != 200:
        table_md, msg, doc_id = refresh_documents(base_url, tenant_id)
        return table_md, msg, doc_id, f"上传失败：{payload.get('message')}"

    data = payload.get("data") or {}
    summary = (data.get("summary") or "").strip()
    detail = (
        f"上传成功\n"
        f"- document_id: `{data.get('document_id')}`\n"
        f"- file_name: {data.get('file_name')}\n"
        f"- chunk_count: {data.get('chunk_count')}\n"
        f"- status: {data.get('status')}\n\n"
        f"**全文摘要**\n{summary or '(无)'}"
    )
    table_md, msg, _ = refresh_documents(base_url, tenant_id)
    doc_id = data.get("document_id") or ""
    return table_md, msg, doc_id, detail


def _iter_sse(response: httpx.Response) -> Generator[tuple[str, dict[str, Any]], None, None]:
    event_name = "message"
    data_lines: list[str] = []

    for raw in response.iter_lines():
        line = raw.rstrip("\r")
        if line == "":
            if data_lines:
                joined = "\n".join(data_lines)
                try:
                    data = json.loads(joined)
                except json.JSONDecodeError:
                    data = {"raw": joined}
                yield event_name, data
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

    if data_lines:
        joined = "\n".join(data_lines)
        try:
            data = json.loads(joined)
        except json.JSONDecodeError:
            data = {"raw": joined}
        yield event_name, data


def chat_stream(
    message: str,
    history: list[dict[str, str]],
    base_url: str,
    tenant_id: str,
    document_id: str | None,
    top_k: int,
    use_bm25: bool,
) -> Generator[tuple[list[dict[str, str]], str, str, str], None, None]:
    question = (message or "").strip()
    history = list(history or [])
    if not question:
        yield history, "请输入问题", "[BM25] 等待提问", "[合并结果] 等待提问"
        return

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "检索中…"})
    yield history, "正在检索…", "正在检索…", "正在检索…"

    body: dict[str, Any] = {
        "question": question,
        "top_k": int(top_k),
        "use_bm25": bool(use_bm25),
    }
    if document_id:
        body["document_id"] = document_id

    dense_text = "暂无结果"
    bm25_text = "[BM25] 暂无结果"
    merged_text = "[合并结果] 暂无结果"
    answer_parts: list[str] = []

    try:
        with httpx.Client(timeout=CHAT_TIMEOUT) as client:
            with client.stream(
                "POST",
                _chat_url(base_url),
                params={"tenant_id": tenant_id or "demo-tenant"},
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                for event, data in _iter_sse(resp):
                    if event == "contexts":
                        dense_text, bm25_text, merged_text = _split_retrieval_views(data)
                        history[-1]["content"] = "检索完成，生成回答中…"
                        yield history, dense_text, bm25_text, merged_text
                    elif event == "delta":
                        chunk = data.get("content") or ""
                        answer_parts.append(chunk)
                        history[-1]["content"] = "".join(answer_parts)
                        yield history, dense_text, bm25_text, merged_text
                    elif event == "error":
                        err = data.get("message") or str(data)
                        history[-1]["content"] = f"错误：{err}"
                        yield history, dense_text, bm25_text, merged_text
                        return
                    elif event == "done":
                        if not answer_parts:
                            history[-1]["content"] = history[-1]["content"] or "(空回答)"
                        yield history, dense_text, bm25_text, merged_text
                        return
    except Exception as exc:
        history[-1]["content"] = f"请求失败：{exc}"
        yield history, dense_text, bm25_text, merged_text
        return

    if not answer_parts and history[-1]["content"].startswith("检索"):
        history[-1]["content"] = "(未收到回答)"
    yield history, dense_text, bm25_text, merged_text


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PDF Agent") as demo:
        gr.Markdown("# PDF Agent\n上传 PDF → 向量入库 → 流式问答（对接本地 FastAPI）")

        with gr.Row():
            base_url = gr.Textbox(label="API Base URL", value=DEFAULT_BASE_URL, scale=3)
            tenant_id = gr.Textbox(label="tenant_id", value="demo-tenant", scale=1)
            user_id = gr.Textbox(label="user_id", value="demo-user", scale=1)

        with gr.Row():
            with gr.Column(scale=1, min_width=420):
                gr.Markdown("### 文档")
                file_input = gr.File(label="上传 PDF", file_types=[".pdf"], type="filepath")
                upload_btn = gr.Button("上传并入库", variant="primary")
                upload_status = gr.Markdown("等待上传…")
                refresh_btn = gr.Button("刷新文档列表")
                list_status = gr.Markdown("")
                document_table = gr.Markdown("暂无文档")
                document_id_input = gr.Textbox(
                    label="限定 document_id（留空=全库检索）",
                    placeholder="从上方表格复制 document_id",
                    lines=2,
                )
                clear_doc_btn = gr.Button("清除文档限定")
                top_k = gr.Slider(1, 20, value=5, step=1, label="top_k")
                use_bm25 = gr.Checkbox(value=True, label="混合 BM25")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话", height=420)
                question = gr.Textbox(
                    label="问题",
                    placeholder="例如：隐睾症的诊断标准是什么？",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

                with gr.Tabs():
                    with gr.Tab("向量检索 dense"):
                        dense_box = gr.Textbox(
                            label="[标签: 向量检索 dense]",
                            value="问答后显示",
                            lines=18,
                            max_lines=30,
                            interactive=False,
                        )
                    with gr.Tab("BM25 检索"):
                        bm25_box = gr.Textbox(
                            label="[标签: BM25]",
                            value="问答后显示",
                            lines=18,
                            max_lines=30,
                            interactive=False,
                        )
                    with gr.Tab("合并结果"):
                        merged_box = gr.Textbox(
                            label="[标签: 合并去重（用于回答）]",
                            value="问答后显示",
                            lines=18,
                            max_lines=30,
                            interactive=False,
                        )

        upload_btn.click(
            fn=upload_document,
            inputs=[file_input, base_url, tenant_id, user_id],
            outputs=[document_table, list_status, document_id_input, upload_status],
        )
        refresh_btn.click(
            fn=refresh_documents,
            inputs=[base_url, tenant_id],
            outputs=[document_table, list_status, document_id_input],
        )
        clear_doc_btn.click(
            fn=lambda: "",
            outputs=[document_id_input],
        )
        send_btn.click(
            fn=chat_stream,
            inputs=[question, chatbot, base_url, tenant_id, document_id_input, top_k, use_bm25],
            outputs=[chatbot, dense_box, bm25_box, merged_box],
        ).then(fn=lambda: "", outputs=[question])
        question.submit(
            fn=chat_stream,
            inputs=[question, chatbot, base_url, tenant_id, document_id_input, top_k, use_bm25],
            outputs=[chatbot, dense_box, bm25_box, merged_box],
        ).then(fn=lambda: "", outputs=[question])
        clear_btn.click(
            fn=lambda: ([], "问答后显示", "问答后显示", "问答后显示"),
            outputs=[chatbot, dense_box, bm25_box, merged_box],
        )

        demo.load(
            fn=refresh_documents,
            inputs=[base_url, tenant_id],
            outputs=[document_table, list_status, document_id_input],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=DEFAULT_GRADIO_PORT,
        share=False,
    )
