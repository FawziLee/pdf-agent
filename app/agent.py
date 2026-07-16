import sys
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC
from dataclasses import dataclass, field

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
