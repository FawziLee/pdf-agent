from typing import List, Dict, Any
from abc import ABC
from dataclasses import dataclass, field

from app.core.llm_engine import LLMEngine, get_llm_engine
from app.core.document_loader import PDFDocumentLoader
from app.schemas.document import OcrParseResult, DocumentSection


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


class PdfAgent(Agent):

    def __init__(self, pdf_path: str = "", document_id: str = ""):
        super().__init__(name="PdfAgent", description="PdfAgent", tools=[], llm_engine=llm_engine)

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

    async def generate_summary(
        self,
        ocr_parse_result: OcrParseResult,
        keep_level: int = 2,
        with_overall: bool = True, # 是否汇总全文
    ) -> DocumentSummary:
        """按二级标题分段总结，并可再汇总全文。"""
        sections = self.build_summary_sections(ocr_parse_result, keep_level=keep_level)
        section_summaries: list[SectionSummary] = []

        for section in sections:
            if not section.text.strip():
                continue
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
