import sys
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator
from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache

# 支持直接运行：python app/agent.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.llm_engine import LLMEngine, get_llm_engine
from app.core.document_loader import PDFDocumentLoader
from app.core.embedding import get_embedding_engine
from app.core.vector_db import get_vector_db_manager
from app.schemas.document import (
    OcrParseResult,
    DocumentSection,
    DocumentContentTree,
    OcrBlock,
)
from loguru import logger


llm_engine = get_llm_engine()


class Agent(ABC):
    def __init__(self, name: str, description: str, tools: List[Dict[str, Any]], llm_engine: LLMEngine):
        self.name = name
        self.description = description
        self.tools = tools
        self.llm_engine = llm_engine


@dataclass
class SectionSummary:
    title: str
    level: int
    summary: str
    page_idxs: list[int] = field(default_factory=list)


@dataclass
class DocumentSummary:
    sections: list[SectionSummary]
    overall: str = ""

    def sections_to_json(self) -> str:
        """分段摘要序列化为 JSON 字符串，便于入库。"""
        import json

        return json.dumps(
            [
                {
                    "title": s.title,
                    "level": s.level,
                    "summary": s.summary,
                    "page_idxs": s.page_idxs,
                }
                for s in self.sections
            ],
            ensure_ascii=False,
        )


class PdfAgent(Agent):

    def __init__(self):
        super().__init__(name="PdfAgent", description="PdfAgent", tools=[], llm_engine=llm_engine)
        self.embedding_engine = get_embedding_engine()
        self.vector_db = get_vector_db_manager()

    async def load_pdf(self, pdf_path: str, document_id: str, document_name: str) -> OcrParseResult:
        pdf_loader = PDFDocumentLoader(
            file_path=pdf_path,
            document_id=document_id,
            document_name=document_name,
        )
        return await pdf_loader.load()

    def build_summary_sections(
        self,
        ocr_parse_result: OcrParseResult,
        keep_level: int = 2,
    ) -> list[DocumentSection]:
        """保留二级及以上标题，下级合并成大文本段。"""
        return ocr_parse_result.content_tree.to_summary_sections(keep_level=keep_level)

    def _blocks_to_chunks(
        self,
        ocr_parse_result: OcrParseResult,
        document_id: str,
        tenant_id: str = "default",
        min_chars: int = 10,
        min_level: int = 2,
        skip_titles: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """从 content_tree 取 level>=min_level 下的 text 块向量化。

        向量化文本 = paragraph_title（节点标题） + 正文 text。
        """
        skip = skip_titles or {"参考文献"}
        chunks: list[dict[str, Any]] = []

        def walk(node: DocumentContentTree) -> None:
            if node.title in skip:
                return

            # 只处理二级及以下标题节点下的正文
            if node.level >= min_level:
                heading = (node.title or "").strip()
                for block in node.content:
                    if not isinstance(block, OcrBlock):
                        continue
                    if block.block_label != "text":
                        continue
                    body = (block.block_content or "").strip()
                    if len(body) < min_chars:
                        continue
                    # 标题拼到正文前一起向量化
                    embed_text = f"{heading}\n{body}" if heading else body
                    chunks.append(
                        {
                            "document_id": document_id,
                            "tenant_id": tenant_id,
                            "page_idx": block.page_idx,
                            "text": embed_text,
                            "bbox": block.block_bbox,
                        }
                    )

            for child in node.children:
                walk(child)

        walk(ocr_parse_result.content_tree)
        return chunks

    async def index_document(
        self,
        ocr_parse_result: OcrParseResult,
        document_id: str,
        tenant_id: str = "default",
        replace_existing: bool = True,
        skip_titles: set[str] | None = None,
    ) -> int:
        """向量化文档块并写入 Milvus。

        规则：content_tree 中 level>=2 节点下、block_label=text 的块；
        向量化时在正文前拼接该节点的 paragraph_title（节点 title）。

        Returns:
            插入条数
        """
        chunks = self._blocks_to_chunks(
            ocr_parse_result,
            document_id,
            tenant_id,
            skip_titles=skip_titles,
        )
        if not chunks:
            logger.warning(f"无可入库文本块：document_id={document_id}")
            return 0

        if replace_existing:
            self.vector_db.delete_by_document_id(document_id, tenant_id)

        texts = [c["text"] for c in chunks]
        vectors = await self.embedding_engine.async_batch_embed(texts)
        count = self.vector_db.insert_chunks(chunks, vectors)
        logger.info(
            f"文档向量化入库完成：document_id={document_id}, tenant={tenant_id}, count={count}"
        )
        return count

    async def generate_summary(
        self,
        ocr_parse_result: OcrParseResult,
        keep_level: int = 2,
        with_overall: bool = True,  # 是否汇总全文
        max_sections: int | None = None,  # 测试时可只跑前 N 节
        skip_titles: set[str] | None = None,  # 如 {"参考文献"}
    ) -> DocumentSummary:
        """按二级标题分段总结，并可再汇总全文。"""
        sections = self.build_summary_sections(ocr_parse_result, keep_level=keep_level)
        skip = skip_titles or set()
        sections = [s for s in sections if s.title not in skip and s.text.strip()]
        if max_sections is not None:
            sections = sections[:max_sections]

        section_summaries: list[SectionSummary] = []
        for section in sections:
            resp = await self.llm_engine.async_chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是文档分析助手。请对给定章节写简洁中文摘要，保留关键结论与要点，不要编造。",
                    },
                    {
                        "role": "user",
                        "content": f"章节标题：{section.title}\n\n正文：\n{section.text[:12000]}",
                    },
                ]
            )
            section_summaries.append(
                SectionSummary(
                    title=section.title,
                    level=section.level,
                    summary=resp.get("content", ""),
                    page_idxs=section.page_idxs,
                )
            )

        overall = ""
        if with_overall and section_summaries:
            joined = "\n\n".join(f"【{s.title}】\n{s.summary}" for s in section_summaries)
            resp = await self.llm_engine.async_chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是文档分析助手。请根据各章节摘要，生成一份全文总摘要。",
                    },
                    {"role": "user", "content": joined[:12000]},
                ]
            )
            overall = resp.get("content", "")

        return DocumentSummary(sections=section_summaries, overall=overall)


    def _dedupe_hits(self, hits: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for hit in hits:
            key = (
                f"{hit.get('document_id')}|{hit.get('page_idx')}|"
                f"{(hit.get('text') or '')[:80]}"
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if limit is not None and len(merged) >= limit:
                break
        return merged

    async def retrieve(
        self,
        question: str,
        tenant_id: str = "default",
        document_ids: list[str] | None = None,
        top_k: int = 5,
        use_bm25: bool = True,
    ) -> dict[str, Any]:
        """检索相关文本块：分别返回稠密向量与 BM25 结果，并给出合并去重后的 contexts。"""
        query = (question or "").strip()
        if not query:
            return {
                "dense": [],
                "bm25": [],
                "contexts": [],
                "bm25_error": None,
                "bm25_enabled": use_bm25,
            }

        query_vec = await self.embedding_engine.async_single_embed(query)
        dense_hits = self.vector_db.search(
            query_vector=query_vec,
            top_k=top_k,
            tenant_id=tenant_id,
            document_ids=document_ids,
        )
        dense = self._dedupe_hits(dense_hits, limit=top_k)

        bm25: list[dict[str, Any]] = []
        bm25_error: str | None = None
        if use_bm25:
            try:
                bm25_hits = self.vector_db.bm25_search(
                    query_text=query,
                    top_k=top_k,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                )
                bm25 = self._dedupe_hits(bm25_hits, limit=top_k)
            except Exception as exc:
                bm25_error = f"{type(exc).__name__}: {exc}"
                logger.warning(f"BM25 检索失败：{bm25_error}")

        contexts = self._dedupe_hits(dense + bm25)
        return {
            "dense": dense,
            "bm25": bm25,
            "contexts": contexts,
            "bm25_error": bm25_error,
            "bm25_enabled": use_bm25,
        }

    async def ask(
        self,
        question: str,
        tenant_id: str = "default",
        document_ids: list[str] | None = None,
        top_k: int = 5,
        use_bm25: bool = True,
    ) -> dict[str, Any]:
        """RAG 问答：先检索上下文，再让 LLM 基于原文回答。"""
        retrieval = await self.retrieve(
            question=question,
            tenant_id=tenant_id,
            document_ids=document_ids,
            top_k=top_k,
            use_bm25=use_bm25,
        )
        contexts = retrieval["contexts"]
        if not contexts:
            return {
                "answer": "未检索到相关文档内容，请先上传并入库 PDF。",
                "dense": retrieval["dense"],
                "bm25": retrieval["bm25"],
                "contexts": [],
                "bm25_error": retrieval.get("bm25_error"),
                "bm25_enabled": retrieval.get("bm25_enabled", use_bm25),
            }

        context_text = "\n\n".join(
            f"[来源 document_id={c.get('document_id')} page={c.get('page_idx')}]\n{c.get('text') or ''}"
            for c in contexts
        )
        resp = await self.llm_engine.async_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是文档问答助手。只根据给定上下文回答，不要编造。"
                        "若上下文不足，明确说明无法从文档中得出结论。"
                        "回答简洁，可引用页码。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n上下文：\n{context_text}",
                },
            ]
        )
        return {
            "answer": resp.get("content", ""),
            "dense": retrieval["dense"],
            "bm25": retrieval["bm25"],
            "contexts": contexts,
            "bm25_error": retrieval.get("bm25_error"),
            "bm25_enabled": retrieval.get("bm25_enabled", use_bm25),
        }

    async def stream_ask(
        self,
        question: str,
        tenant_id: str = "default",
        document_ids: list[str] | None = None,
        top_k: int = 5,
        use_bm25: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式 RAG 问答：先返回检索块，再流式返回回答。"""
        retrieval = await self.retrieve(
            question=question,
            tenant_id=tenant_id,
            document_ids=document_ids,
            top_k=top_k,
            use_bm25=use_bm25,
        )
        contexts = retrieval["contexts"]
        yield {"event": "contexts", "data": retrieval}

        if not contexts:
            message = "未检索到相关文档内容，请先上传并入库 PDF。"
            yield {"event": "delta", "data": {"content": message}}
            yield {"event": "done", "data": {"finish_reason": "no_context"}}
            return

        context_text = "\n\n".join(
            f"[来源 document_id={c.get('document_id')} page={c.get('page_idx')}]\n"
            f"{c.get('text') or ''}"
            for c in contexts
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是文档问答助手。只根据给定上下文回答，不要编造。"
                    "若上下文不足，明确说明无法从文档中得出结论。"
                    "回答简洁，可引用页码。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n上下文：\n{context_text}",
            },
        ]

        async for chunk in self.llm_engine.async_stream_chat(messages=messages):
            yield {"event": "delta", "data": {"content": chunk}}
        yield {"event": "done", "data": {"finish_reason": "stop"}}


@lru_cache(maxsize=1)
def get_pdf_agent() -> PdfAgent:
    return PdfAgent()

if __name__ == "__main__":
    import asyncio
    import json
    from pathlib import Path

    from app.core.ocr_tools import OCR_RESULT_PATH, PaddleOcrTool
    from app.schemas.document import OcrParseResult

    async def main():
        # 用已有 OCR 结果，不重新跑 Paddle
        ocr = PaddleOcrTool(document_name="隐睾症")
        merged = json.loads((OCR_RESULT_PATH / "隐睾症.json").read_text(encoding="utf-8"))
        extracted = ocr._extract_result(merged)
        tree = ocr._blocks_to_tree(extracted)
        parse_result = OcrParseResult(
            total_pages=len(merged.get("layoutParsingResults", [])),
            pages=[],
            result=merged,
            extracted_result=extracted,
            content_tree=tree,
        )

        agent = PdfAgent()
        # 先测前 3 节 + 总摘要，跳过参考文献（省 token / 时间）
        summary = await agent.generate_summary(
            parse_result,
            keep_level=2,
            with_overall=True,
            max_sections=3,
            skip_titles={"参考文献"},
        )

        print("===== 章节摘要 =====")
        for s in summary.sections:
            print(f"\n【{s.title}】 pages={s.page_idxs}\n{s.summary}")
        print("\n===== 全文总摘要 =====\n")
        print(summary.overall)

        out = {
            "overall": summary.overall,
            "sections": [
                {
                    "title": s.title,
                    "level": s.level,
                    "page_idxs": s.page_idxs,
                    "summary": s.summary,
                }
                for s in summary.sections
            ],
        }
        out_path = OCR_RESULT_PATH / "隐睾症_llm_summary.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存: {out_path}")

    asyncio.run(main())
